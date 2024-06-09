import cv2
import torch
import numpy as np

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from utils.detect_pose import processed_image

MODEL_WEIGHT='checkpoint_iter_370000.pth'

net = PoseEstimationWithMobileNet().cuda()
checkpoint = torch.load(MODEL_WEIGHT, map_location=torch.device('cuda:0'))
load_state(net, checkpoint)


import pickle

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
'''model for pose classification'''
__pose_clf = PoseClassifier(
    "./CaptureAdaBoostClassifier.pkl",
    "./LandAdaBoostclassifier.pkl"
)
count_1 = 0
count_2 = 0
cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    poses,_=processed_image(net,frame)
    keypoints = np.array(poses[0].keypoints).flatten() if len(poses)>0 else np.zeros(36,)
    keypoints=keypoints.reshape(1,-1)
    cv2.imshow('frame', frame)
    # 1 take photo, 2 land , 0 others
    if __pose_clf.predict(keypoints) == 1:
        count_1 += 1
        count_2=0
    elif __pose_clf.predict(keypoints) == 2:
        count_2 += 1
        count_1=0
    if count_1 == 5:
        print("Capture")
        count_1 = 0
        count_2 = 0
    if count_2 == 5:
        print("Land")
        count_1 = 0
        count_2 = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break