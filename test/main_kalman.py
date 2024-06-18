import time
import sys
import tellopy
import keyboard
import cv2
import av
import threading
import traceback
from simple_pid import PID
from pygame.locals import *
import numpy as np
import time
import cv2
import torch
import time

from utils import kalman
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from modules.detect_pose import processed_image
from modules.detect_pose import caculate_distance
DEVICE = 'cuda:0'
MODEL_WEIGHT='checkpoint_iter_370000.pth'
# VIDEO_SIZE=256


# prev_flight_data = None
run_controller_thread = True
shutdown = False

#drone control inputs

# tracking options
use_vertical_tracking = True
use_rotation_tracking = True
use_horizontal_tracking = True
use_distance_tracking = True

dist_setpoint = 100
area_setpoint = 13
cx = 0
cy = 0

# Kalman estimator scale factors
kvscale = 6
khscale = 4
distscale = 3

vx,dist,vy,rx = 0,0,0,0


def safety_limiter(leftright,fwdbackw,updown,yaw, SAFETYLIMIT=30):
    """
    Implement a safety limiter if values exceed defined threshold

    Args:
        leftright ([type]): control value for left right
        fwdbackw ([type]): control value for forward backward
        updown ([type]): control value up down
        yaw ([type]): control value rotation
    """
    val = np.array([leftright,fwdbackw,updown,yaw])

    # test uppler lover levels
    val[val>=SAFETYLIMIT] = SAFETYLIMIT
    val[val<=-SAFETYLIMIT] = -SAFETYLIMIT

    return val[0],val[1],val[2],val[3]

def controller_thread():
    global drone
    global shutdown
    global run_controller_thread
    global vx
    global dist
    global vy
    global rx

    control_on = True #allows you to toggle control so that you can force landing

    print('start controller_thread()')
    try:
        while run_controller_thread:
            time.sleep(.05)
            # takeoff
            if keyboard.is_pressed('space'):
                drone.takeoff()
            # land
            elif keyboard.is_pressed('l'):
                drone.land()
                control_on = False #disable control
                shutdown = True
            elif keyboard.is_pressed('q'):
                drone.counter_clockwise(40)
            elif keyboard.is_pressed('e'):
                drone.clockwise(40)
            elif keyboard.is_pressed('d'):
                drone.right(40)
            elif keyboard.is_pressed('a'):
                drone.left(40)
            elif keyboard.is_pressed('w'):
                drone.forward(40)
            elif keyboard.is_pressed('s'):
                drone.backward(40)
            elif keyboard.is_pressed('r'):
                drone.clockwise(0)
                drone.forward(0)
                drone.left(0)
            elif keyboard.is_pressed('t'): #toggle controls
                if control_on:
                    control_on = False
                else:
                    control_on = True
            elif keyboard.is_pressed('esc'):
                drone.land()
                break

            #set commands based on kalman filter
            #  TODO
            if control_on :
                if rx < 0:
                    drone.clockwise(int(rx)*-1)
                else:
                    drone.counter_clockwise(int(rx))
                
                if vy < 0:
                    drone.down(int(vy)*-1) #easily moving downwards requires control output to be magnified
                else:
                    drone.up(int(vy))
                
                if dist < 0:
                    drone.forward(-dist) #easily moving downwards requires control output to be magnified
                else:
                    drone.forward(dist)
                
            

    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(e)
    finally:
        run_controller_thread = False

# def handler(event, sender, data, **args):
#     global prev_flight_data
#     drone = sender
#     if event is drone.EVENT_FLIGHT_DATA:
#         if prev_flight_data != str(data):
#             print(data)
#             prev_flight_data = str(data)
#     else:
#         print('event="%s" data=%s' % (event.getname(), str(data)))

def main():
    global vx
    global dist
    global vy
    global rx

    drone = tellopy.Tello()
    drone.connect()
    drone.wait_for_connection(60.0)
    drone.start_video()

    # drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)


    kf = kalman.clKalman()
    kfarea= kalman.clKalman()


    print("Start Running")

    # Load model
    print("Loading model")
    net = PoseEstimationWithMobileNet().cuda()
    checkpoint = torch.load(MODEL_WEIGHT, map_location=torch.device(DEVICE))
    load_state(net, checkpoint)

    try:
        # threading.Thread(target=recv_thread).start()
        threading.Thread(target=controller_thread).start()
        
        container = av.open(drone.get_video_stream())

        frame_count = 0
        while not shutdown:
            for frame in container.decode(video=0):
                frame_count = frame_count + 1
                # skip first 300 frames
                if frame_count < 300:
                    continue
                if frame_count %4 == 0:
                    frame_data = frame.to_ndarray(format='bgr24').astype('uint8')
                    current_poses,overlay_image = processed_image(net,frame_data)
                    
                    if len(current_poses) > 0:
                        # dist = 0
                        # vy = 0
                        # vx,rx = 0,0
                        pose=current_poses[0]
                        pose_nose = pose.keypoints[0]

                        kf.init(cx,cy)

                        # compute init 'area', ignor x dimension
                        kfarea.init(1,pose_nose[1])

                        # process corrections, compute delta between two objects
                        cp = kf.predictAndUpdate(cx,cy)

                        # calculate delta over 2 axis
                        mvx = -int((cp[0]-pose_nose[0])//kvscale)
                        mvy = int((cp[1]-pose_nose[1])//khscale)

                        # estimate object distance
                        area_frame = overlay_image.shape[0]*overlay_image.shape[1]
                        area_det = caculate_distance(pose.keypoints[0],pose.keypoints[1])**2
                        area_ratio =int((area_det/area_frame)*1000) #TODO
                        _, ocp = kfarea.predictAndUpdate(1, area_ratio, True)

                        dist = int((ocp[1]-dist_setpoint)//distscale)
                        
                        # don't combine horizontal and rotation
                        if use_horizontal_tracking:
                            rx = 0
                            vx = mvx
                        if use_rotation_tracking:
                            vx = 0
                            rx = mvx
                        if use_vertical_tracking:
                            vy = mvy

                        # limit signals if is the case, could save your tello
                        vx,dist,vy,rx = safety_limiter(vx,dist,vy,rx,SAFETYLIMIT=40)

                    else:
                        vx,dist,vy,rx = 0,0,0,0
                        cv2.imshow('Tello Video Stream', overlay_image)
                        cv2.waitKey(1) 
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(e)

    cv2.destroyAllWindows()
    drone.quit()
    exit(1)


if __name__ == '__main__':
    main()