
# Features
1.  Manual Control: Control the drone manually using keyboard inputs.
2.  Automatic Control: Enable automatic control using PID outputs.
3.  Pose Control: Trigger actions based on specific poses detected.

# Getting Started
#### Environment
* Python 3.x
* CUDA 11.x
*  Train a model follow [this](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
* `pip install -r requirements.txt`
# Run
1.  Open Tello or Tello edu
2.  Connect Tello with wifi
3.  `python ./main.py`

# Control
### Manual Control
##### Use the following keyboard inputs to manually control the drone:
* `t`: Toggle between manual and automatic control.
* `space`: Takeoff.
* `L`:  Land.
* `Q`:  Rotate counter-clockwise.
* `E`:  Rotate clockwise.
* `D`:  Move right.
* `A`:  Move left.
* `W`:  Move forward.
* `S`:  Move backward.
* `R`:  Stop all movement.
* `ESC`:  Land safely and shut down the system.


### Automatic Control
##### When automatic control is enabled (toggled by `t`), the drone's movements are controlled by PID outputs

##### The system adjusts the drone's rotation, forward/backward movement, and up/down movement based on the PID controller outputs.

### Pose Control
##### The system continuously monitors for specific poses when auto_control is enabled. If a specific pose(rise your right hand) is detected for 5 consecutive frames, the drone will land safely.
   


# Human Pose keypoints reflect

<div class="center">
<table>
<tr>
 <td>

| number | part |
|---|---|
|  0 | nose  |
|  1 | neck  |
|  2 | r-shoulder  |
|  3 | r-elbow  |
|  4 | r-wrist  |
|  5 | l-shoulder  |
|  6 | l-elbow  |
|  7 |  l-wrist |
|  8 | r-hip  |
|  9 | r-knee  |
|  10 | r-ankle  |
|  11 | l-hip   |
|  12 | l-knee  |
|  13 | l-ankle  |
|  14 | r-eye  |
|  15 | l-eye  |
|  16 | r-ear  |
|  17 | l-ear  |
 </td>

 <td>

![6aa8621b20e341139a0e8be3d9470364](https://github.com/We51ey/tello/assets/161515320/4a7856c3-ff4f-40e6-beec-b09fca2cf957)
 </td>
</tr>
</table>
 </div>




## Ref
*  https://github.com/fvilmos/tello_object_tracking
*  https://gobot.io/blog/2018/04/20/hello-tello-hacking-drones-with-go/
*  https://pkg.go.dev/gobot.io/x/gobot/platforms/dji/tello#Driver
*  https://dl.djicdn.com/downloads/RoboMaster%20TT/Tello_SDK_3.0_User_Guide.pdf
*  https://github.com/Matthewjsiv/Person-Tracking-Tello-Drone
*  https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

