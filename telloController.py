import time
import sys
import tellopy
import keyboard
import cv2
import av
import threading
import traceback
import torch
import pickle
import numpy as np
from simple_pid import PID
from pygame.locals import *
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from utils.detect_pose import processed_image,caculate_distance

class PoseClassifier:
    def __init__(self, capture_pkl, land_pkl):
        self.capture = pickle.load(open(capture_pkl, 'rb'))
        self.land = pickle.load(open(land_pkl, 'rb'))

    def predict(self, x) -> int: # 1 for capture, 2 for land and 0 for others
        if self.capture.predict(x) == 1:
            return 1
        elif self.land.predict(x) == 1:
            return 2
        return 0

'''This class is used to control the drone using the pose of the person'''
class TelloController(object):
    def __init__(self):
        self.DEVICE = 'cuda:0'
        self.MODEL_WEIGHT='weights/checkpoint_iter_370000.pth'
        self.net=None
        self.checkpoint=None

        self.run_controller_thread = True
        self.shutdown = False
        self.control_on = False

        self.drone_cc = 0
        self.drone_ud = 0
        self.drone_fb = 0

        self.pdrone_cc = -111
        self.pdrone_ud = -111
        self.pdrone_fb = -111

        self.current_poses =[]
        self.overlay_image=None
        self.screen_center=[0,0]
        self.pose=None  #pose of the person
        self.pose_center=None 

        self.frame_provider=None

        self.drone=tellopy.Tello()
        self.pose_clf = PoseClassifier(
            "classifier/CaptureAdaBoostClassifier.pkl",
            "classifier/LandAdaBoostclassifier.pkl"
        )

        # pid_cc = PID(0.35,0.1,0.35,setpoint=0,output_limits=(-50,50))
        self.pid_cc = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))

        # pid_ud = PID(0.3,0.3,0.3,setpoint=0,output_limits=(-40,40))
        self.pid_ud = PID(0.3,0,0.01,setpoint=0,output_limits=(-40,40))

        # pid_fb = PID(0.35,0.1,0.35,setpoint=0,output_limits=(-50,50))
        # self.pid_fb = PID(0.5,0.04,0.3,setpoint=0,output_limits=(-50,50))
        self.pid_fb = PID(0.5,0.04,0.05,setpoint=0,output_limits=(-50,50))

    def __update_pid(self):

        if self.pose is not None:
            desiredHeight = 150
            # caculate the height of the person
            leftSholy = self.pose.keypoints[5]
            rightSholy = self.pose.keypoints[2]
            meanHeight = caculate_distance(leftSholy,rightSholy)

            error_fb = meanHeight - desiredHeight
            error_x=self.pose_center[0]-self.screen_center[0]
            error_y=self.pose_center[1]-self.screen_center[1]

            self.drone_cc = self.pid_cc(error_x) if abs(error_x) > 60 else 0
            self.drone_ud = self.pid_ud(error_y) if abs(error_y) > 90 else 0
            self.drone_fb = self.pid_fb(error_fb) if abs(error_fb) > 10 else 0
        else:
            self.drone_cc = 0
            self.drone_ud = 0
            self.drone_fb = 0
            self.pid_cc.reset()
            self.pid_ud.reset()
            self.pid_fb.reset()

    def __controller_thread(self):

        try:
            while self.run_controller_thread:
                time.sleep(.05)
                # takeoff
                if keyboard.is_pressed('space'):
                    self.drone.takeoff()
                    self.control_on=True
                    self.drone.up(100)
                # land
                elif keyboard.is_pressed('l'):
                    self.drone.land()
                    self.control_on = False #disable control
                elif keyboard.is_pressed('q'):
                    self.drone.counter_clockwise(40)
                elif keyboard.is_pressed('e'):
                    self.drone.clockwise(40)
                elif keyboard.is_pressed('d'):
                    self.drone.right(40)
                elif keyboard.is_pressed('a'):
                    self.drone.left(40)
                elif keyboard.is_pressed('w'):
                    self.drone.forward(40)
                elif keyboard.is_pressed('s'):
                    self.drone.backward(40)
                elif keyboard.is_pressed('r'):
                    self.drone.clockwise(0)
                    self.drone.forward(0)
                    self.drone.left(0)
                elif keyboard.is_pressed('t'): #toggle controls
                    if self.control_on:
                        self.control_on = False
                    else:
                        self.control_on = True
                elif keyboard.is_pressed('esc'):
                    self.drone.land()
                    self.run_controller_thread = False
                    self.shutdown = True
                    print(self.shutdown)
                    break

                #set commands based on PID output
    ####################################### TODO change the error threshold ########################################

                if self.control_on and (self.pdrone_cc != self.drone_cc):
                    if self.drone_cc < 0:
                        self.drone.clockwise(int(self.drone_cc)*-1)
                    else:
                        self.drone.counter_clockwise(int(self.drone_cc))
                    self.pdrone_cc = self.drone_cc
                if self.control_on and (self.pdrone_fb != self.drone_fb):
                    if self.drone_fb < 0:
                        self.drone.backward(min([50,int(self.drone_fb)*-1])) #easily moving downwards requires control output to be magnified
                    else:
                        self.drone.forward(min([50,int(self.drone_fb)]))
                    self.pdrone_fb = self.drone_fb
                if self.control_on and (self.pdrone_ud != self.drone_ud):
                    if self.drone_ud < 0:
                        self.drone.down(min([100,int(self.drone_ud)*-1])) #easily moving downwards requires control output to be magnified
                    else:
                        self.drone.up(int(self.drone_ud))
                    self.pdrone_ud = self.drone_ud
                
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print(e)
        finally:
            self.run_controller_thread = False
            self.shutdown = True

    def __pose_control(self):
        while not self.shutdown:
            time.sleep(0.3)
            keypoints = np.array(self.pose.keypoints).flatten() if self.pose!=None else np.zeros(36,)
            keypoints = keypoints.reshape(1,-1)

            if self.pose_clf.predict(keypoints) == 2 and self.pose!=None:
                if self.control_on:
                    self.drone.land()
                    self.control_on = False

    def tracking(self):
        # Load the model
        print("Loading model")
        self.net = PoseEstimationWithMobileNet().cuda()
        self.checkpoint = torch.load(self.MODEL_WEIGHT, map_location=torch.device(self.DEVICE))
        load_state(self.net, self.checkpoint)
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        self.drone.start_video()
        # record video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/output.avi',fourcc,20,(960,720),True)

        threading.Thread(target=self.__controller_thread).start()
        threading.Thread(target=self.__pose_control).start()

        try:
            self.frame_provider = av.open(self.drone.get_video_stream())
            frame_skip = 0
            while not self.shutdown:
                for frame in self.frame_provider.decode(video=0):
                    frame_skip = frame_skip + 1
                    if frame_skip < 300: # skip first 300 frames
                        continue
                    if frame_skip %2 == 0:
                        frame_data = frame.to_ndarray(format='bgr24').astype('uint8')
                        self.current_poses,self.overlay_image = processed_image(self.net,frame_data,self.current_poses)
                        self.screen_center=[self.overlay_image.shape[1]//2,self.overlay_image.shape[0]//2]
                        if len(self.current_poses) >  0:
                            self.pose=self.current_poses[0]
                            self.pose_center = self.pose.keypoints[1]
                            self.overlay_image = cv2.line(self.overlay_image, (self.screen_center[0], self.screen_center[1]), (self.pose_center[0],self.pose_center[1]-10), (255, 255, 0), 2)
                        else:
                            self.pose=None
                        out.write(self.overlay_image)
                        self.__update_pid()
                        cv2.imshow('Tello Video Stream', self.overlay_image)
                        cv2.waitKey(1)
            out.release()
            cv2.destroyAllWindows()
            self.drone.quit()
            exit(1)  
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print(e)
        finally:
            # self.is_recording = False
            out.release()
            cv2.destroyAllWindows()
            threading.Thread(target=self.__controller_thread).start()
            threading.Thread(target=self.__pose_control).start()
            self.drone.quit()
            exit(1)

if __name__ == '__main__':
    telloController =TelloController()
    telloController.tracking()