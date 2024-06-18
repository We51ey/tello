import time
import sys
import tellopy
import keyboard
import cv2
import av
import threading
import traceback
import torch
from simple_pid import PID
from pygame.locals import *

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.detect_pose import processed_image

DEVICE = 'cuda:0'
MODEL_WEIGHT='checkpoint_iter_370000.pth'
TELLO_ADDRESS = ('192.168.10.1',8889)
VIDEO_SIZE=256


run_controller_thread = True
shutdown = False
# video_frame_queue=queue.Queue(32)
#drone control inputs
drone_cc = 0
drone_ud = 0
drone_fb = 0


def controller_thread():
    global drone
    global drone_cc
    global drone_ud
    global drone_fb
    global shutdown
    #initialize previous drone control inputs
    control_on = True #allows you to toggle control so that you can force landing
    pdrone_cc = -111
    pdrone_ud = -111
    pdrone_fb = -111

    global run_controller_thread
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

            #set commands based on PID output
####################################### TODO change the error threshold ########################################
            if control_on and (pdrone_cc != drone_cc):
                if drone_cc < 0:
                    drone.clockwise(int(drone_cc)*-1)
                else:
                    drone.counter_clockwise(int(drone_cc))
                pdrone_cc = drone_cc
            if control_on and (pdrone_ud != drone_ud):
                if drone_ud < 0:
                    drone.down(min([100,int(drone_ud)*-1])) #easily moving downwards requires control output to be magnified
                else:
                    drone.up(int(drone_ud))
                pdrone_ud = drone_ud
            if control_on and (pdrone_fb != drone_fb):
                if drone_fb < 0:
                    drone.backward(min([50,int(drone_fb)*-1])) #easily moving downwards requires control output to be magnified
                else:
                    drone.forward(min([50,int(drone_fb)]))
                pdrone_fb = drone_fb

    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(e)
    finally:
        run_controller_thread = False
        # video_frame_queue.put(None)
        # video_frame_queue.queue.clear()

# def handler(event, sender, data, **args):
#     global video_frame_queue
#     drone = sender
#     if event is drone.EVENT_VIDEO_FRAME:
#         if video_frame_queue.full():
#             video_frame_queue.get()
#     else:
#         video_frame_queue.put(data)

# def read_video_frame(video_frame_queue, timeout=3):
#     try:
#         latest_video_frame = None
#         while True:
#             video_frame = video_frame_queue.get(timeout=timeout)
#             if video_frame is not None:
#                 latest_video_frame = video_frame
#     except queue.Empty:
#         return latest_video_frame

def get_latest_video_frame(frame_provider):
    frame = None
    try:
        for frame_buffer in frame_provider:
            frame = frame_buffer
    except StopIteration as e:
        frame = None
    return frame


def main():
    global drone
    global drone_cc
    global drone_ud
    global drone_fb
    global shutdown
    drone = tellopy.Tello()
    drone.connect()
    drone.wait_for_connection(60.0)
    drone.start_video()
    ############################################################## TODO: change the PID values ##############################################################
    pid_cc = PID(0.35,0.2,0.2,setpoint=0,output_limits=(-50,50))
    pid_ud = PID(0.3,0.3,0.3,setpoint=0,output_limits=(-40,40))
    pid_fb = PID(0.3,0.1,0.4,setpoint=0,output_limits=(-10,10))
    ############################################################## TODO: change the PID values ##############################################################

    # Subscribe to the video frame event
    # drone.subscribe(drone.EVENT_VIDEO_FRAME,handler)
    print("Start Running")

    # Load model
    print("Loading model")
    net = PoseEstimationWithMobileNet().cuda()
    checkpoint = torch.load(MODEL_WEIGHT, map_location=torch.device(DEVICE))
    load_state(net, checkpoint)

    try:
        threading.Thread(target=controller_thread).start()
        frame_provider = av.open(drone.get_video_stream())

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc,20,(960,720),True)
        frame_count = 0
        while not shutdown:
           for frame in frame_provider.decode(video=0):
                frame_count = frame_count + 1
                # skip first 300 frames
                if frame_count < 300:
                    continue
                if frame_count %2 == 0:
                    frame_data = frame.to_ndarray(format='bgr24').astype('uint8')
                    current_poses,overlay_image = processed_image(net,frame_data)

                    screen_center=[overlay_image.shape[1]//2,overlay_image.shape[0]//2]
                    if len(current_poses) >  0:
                        pose=current_poses[0]
                        pose_nose = pose.keypoints[0]

                        overlay_image = cv2.line(overlay_image, (screen_center[0], screen_center[1]), (pose_nose[0],pose_nose[1]-10), (255, 255, 0), 2)

                        ctrl_out_cc = 0
                        ctrl_out_ud = 0
                        
                        errorx = 0
                        errory = 0
                    
                        errorx=pose_nose[0]-screen_center[0]
                        errory=pose_nose[1]-screen_center[1]

                        # control the roll of the drone
                        if abs(errorx) > 60:
                            ctrl_out_cc = pid_cc(errorx)
                            drone_cc = ctrl_out_cc
                        else:
                            drone_cc = 0
                            # TODO

                        if abs(errory) > 90:
                            ctrl_out_ud = pid_ud(errory)
                            drone_ud = ctrl_out_ud
                        else:
                            drone_ud = 0


                        desiredHeight = 200

                        # determine if the hips and shoulders are in the frame
                        if pose.keypoints[11][0] > -1 and pose.keypoints[8][0] > -1:
                            # caculate the height of the person
                            leftSholy = int(pose.keypoints[5][0])
                            rightSholy = int(pose.keypoints[2][0])
                            leftHipy = int(pose.keypoints[11][0])
                            rightHipy = int(pose.keypoints[8][0])
                            meanHeight = ((rightHipy - rightSholy) + (leftHipy - leftSholy))/2 #technically arbitrary
                            errorFB = meanHeight - desiredHeight
                            #error can be within +/- 15 without caring
                            if abs(errorFB) > 15:
                                ctrl_out_fb = pid_cc(errorFB)
                                drone_fb = ctrl_out_fb
                            else:
                                drone_fb = 0
                        else:
                            #reset pid
                            drone_fb = 0
                            pid_fb.reset()

                    else:
                        drone_cc = 0
                        drone_ud = 0
                        drone_fb = 0
                        pid_cc.reset()
                        pid_ud.reset()
                        pid_fb.reset()
                    cv2.imshow('Tello Video Stream', overlay_image)
                    out.write(overlay_image)
                    cv2.waitKey(1) 
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(e)
    finally:
        out.release()
        cv2.destroyAllWindows()
        drone.quit()
        exit(1)

if __name__ == '__main__':
    main()
