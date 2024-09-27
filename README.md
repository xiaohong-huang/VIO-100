# VIO-100




## 1. Prerequisites
### 1.1 C++14 Compiler
This package requires some features of C++14.

### 1.2 ROS
This package is developed and tested under [ROS Melodic (Ubuntu 18.04)](http://wiki.ros.org/melodic) or [ROS Noetic (Ubuntu 20.04)](http://wiki.ros.org/noetic) environment.


## 2. Build VIO-100
Clone the repository to your catkin workspace (for example `~/catkin_ws/`):
```
cd ~/catkin_ws/src/
git clone https://github.com/xiaohong-huang/VIO-100.git
```
Build the OpenCV4 (>=4.3.0):
```
#Clone the Opencv to the folder
cd ~/catkin_ws/src/VIO-100
git clone https://github.com/opencv/opencv.git
cd opencv

#Switch the blanch to OpenCV 4.10.0. 
git checkout 4.10.0

#build opencv
mkdir build
cd build
cmake ..
make -j8
#Do not use "make install"
```
Note, the OpenCV version must be larger than 4.3.0 (the newest version is 4.10.0, which is work well in our project). Otherwise, some of the function may not work well.

Then build the package with:
```
cd ~/catkin_ws/
catkin_make
```


## 3. Run VIO-100 with EUROC, TUM-VI and Our dataset
Download the EUROC bag [Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

Download the TUM-VI (512x512) Bag [Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset). 

Download Our [Dataset](https://1drv.ms/f/s!ApdCy_pJvU0qyVsLB906CNjAEQiH).

Download the 4Seasons Dataset using the [dm-vio-python-tools](https://github.com/lukasvst/dm-vio-python-tools) and follow the instructions in [4seasons_bag_generate](https://github.com/xiaohong-huang/VIO-100/blob/main/4seasons_bag_generate) to generate the ROS bag format.





Launching the rviz via:
```
source ~/catkin_ws/devel/setup.bash
roslaunch vio_100 visual_inertial_rviz.launch
```
Open another terminal and run the project by:
```
source ~/catkin_ws/devel/setup.bash
rosrun vio_100 vio_100_node src/VIO-100/yaml/SETTING.yaml YOUR_BAG_FOLDER/BAG_NAME.bag ourput.csv
```
YOUR_BAG_FOLDER is the folder where you save the dataset. 
BAG_NAME is the name of the dataset. 
SETTING.yaml is the setting for different datasets. 

You could use the following settings to perform VIO in different datasets.
```
euroc_config.yaml       #EUROC dataset
tum_config.yaml         #TUM-VI dataset
my_config.yaml          #Our dataset
4seasons_config         #4Seasons dataset
```
 We have also provide demos for runing and evaluating with all the datasets (see [run_evaluate_all](https://github.com/xiaohong-huang/VIO-100/blob/main/run_evaluate_all)).  


## 4. Acknowledgements
The VIO framework is developed from [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono). The Dogleg solving strategy is developed base on [Ceres-Solver](http://ceres-solver.org/). The packages for interfacing ROS with OpenCV is forked from [ros-perception](https://github.com/ros-perception/vision_opencv). 

The version with code comments will be uploaded in the future.


## 5. Licence
The source code is released under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.
