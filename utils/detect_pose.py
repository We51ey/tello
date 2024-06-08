import socket
import time
import cv2
import numpy as np
import torch
import time
import math
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses

DEVICE = 'cuda:0'
MODEL_WEIGHT='checkpoint_iter_370000.pth'
TELLO_ADDRESS = ('192.168.10.1',8889)
VIDEO_SIZE=256
'''
0'nose': 鼻子
1'neck': 颈部
2'r_sho': 右肩 (右肩部)
3'r_elb': 右肘 (右肘部)
4'r_wri': 右腕 (右手腕)
5'l_sho': 左肩 (左肩部)
6'l_elb': 左肘 (左肘部)
7'l_wri': 左腕 (左手腕)
8'r_hip': 右臀部 (右髋部)
9'r_knee': 右膝盖 (右膝关节)
10'r_ank': 右脚踝 (右脚踝部)
11'l_hip': 左臀部 (左髋部)
12'l_knee': 左膝盖 (左膝关节)
13'l_ank': 左脚踝 (左脚踝部)
14'r_eye': 右眼
15'l_eye': 左眼
16'r_ear': 右耳
17'l_ear': 左耳
'''
def caculate_distance(point_1,point_2):
	distance = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)
	return distance


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

# 
def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def processed_image(net, img, previous_poses):
	height_size = 256
	cpu = False # use GPU
	track = 1
	smooth = 1
	upsample_ratio = 4
	stride = 8
	num_keypoints = 18
	orig_img = img.copy()
	heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

	total_keypoints_num = 0
	all_keypoints_by_type = []
	for kpt_idx in range(num_keypoints):  # 19th for bg
		total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

	pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
	for kpt_id in range(all_keypoints.shape[0]):
		all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
		all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
	current_poses = []

	for n in range(len(pose_entries)):
		if len(pose_entries[n]) == 0:
			continue
		pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
		for kpt_id in range(num_keypoints):
			if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
				pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
				pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
		pose = Pose(pose_keypoints, pose_entries[n][18])
		if pose.keypoints[0][0]!=-1 and pose.keypoints[5][0] !=-1 and pose.keypoints[2][0] !=-1:
			current_poses.append(pose)

	if track:
		track_poses(previous_poses, current_poses, smooth=smooth)
		previous_poses = current_poses
	for pose in current_poses:
		pose.draw(img)
	img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
	# for pose in current_poses:
	# 	cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
	# 				  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
		# if track:
		# 	cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
		# 				cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

	return current_poses,img