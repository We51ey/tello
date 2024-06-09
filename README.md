
## Environment
* Python 3.9
* CUDA 11.6
## Prepare
1.  `pip install -r requirements.txt`
2.  Train a model follow [this](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
## Run
1.  Open Tello or tello edu
2.  Connect Tello with wifi
3.  `python ./main.py`

## Human Pose keypoints reflect
  neck 颈部、脖子 1  
  l-shoulder 左肩膀 5  
  r-shoulder 右肩膀 2  
l-elbow 左胳膊肘 6  
r-elbow 右胳膊肘 3  
l-wrist 左手腕 7  
r-wrist 右手腕 4  
r-hip 右臀部、髋部 8  
l-hip 左臀部、髋部 11  
r-knee 右膝盖 9  
l-knee 左膝盖 12  
r-ankle 右脚踝 10  
l-ankle 左脚踝 13  
nose 鼻子 0  
l-ear 左耳 17  
r-ear 右耳 16  
l-eye 左眼 15  
r-eye 右眼 14  
![6aa8621b20e341139a0e8be3d9470364](https://github.com/We51ey/tello/assets/161515320/4a7856c3-ff4f-40e6-beec-b09fca2cf957)

## Ref
*  https://github.com/fvilmos/tello_object_tracking
*  https://gobot.io/blog/2018/04/20/hello-tello-hacking-drones-with-go/
*  https://pkg.go.dev/gobot.io/x/gobot/platforms/dji/tello#Driver
*  https://dl.djicdn.com/downloads/RoboMaster%20TT/Tello_SDK_3.0_User_Guide.pdf
*  https://github.com/Matthewjsiv/Person-Tracking-Tello-Drone
*  https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

