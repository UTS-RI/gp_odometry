# GP Distance Field Odometry

The code present in this repository is an application of the accurate distance field based on Gaussian Processes (GPs) presented [here](https://arxiv.org/abs/2302.13005).
Using RGB-D scans from a depth camera, the camera pose is estimated using solely geometric information.
The map of the environment is built/maintained in a sparse voxel grid with naive fusion (centroid of the points in each cell) and simplistic pruning rules.
Each scan is registered in a scan-to-map fashion.
The registration consists of minimising the queried distance values of a new scan in the map's distance field similarly to [LogGPIS-MOP](https://ieeexplore.ieee.org/abstract/document/10202666).
Please note that this package does not reflect the performance of LogGPIS-MOP as the present package does not perform proper fusion, graph optimisation, etc.

This package is based on ROS and can run in real-time in some situations (using a GPU in a small scene).
We will later introduce scripts to process the data in an offline manner.

At the time of writing, this package has only been tested on __Ubuntu 20.04__ with __ROS Noetic__.


If you are using this work please cite the corresponding papers as explained at the bottom of this page.






https://github.com/UTS-CAS/gp_odometry/assets/18108165/3bbaa939-a780-4180-a682-69182f5b7a92








### Dependencies

First, you will need to have ROS1 installed on your machine.

The code has been designed to be usable both on CPU and GPU based on pykeops (included in the requirement file mentioned later).
For optimal performance, you will need an Nvidia GPU with Cuda >10 (and it also requires a standard C++ compiler to be installed on the machine, c.f. pykeops' documentation).

The Python packages needed are listed in `requirements.txt` (installation instructions in the next section).

### Install

Clone this repository in the sources of your catkin workspace (e.g. `~/catkin_ws/src/`, if does not exist create a catkin workspace as per the ROS tutorials).

Install the dependencies with
```
pip install -r requirements.txt
```

Then use `catkin_make` or `catkin build gp_odometry` depending on your setup.

### Use

The package provides a convenient launch file that only uses a handful of parameters:
```
roslaunch gp_odometry gp_odometry.launch PARAM1:=VALUE1 PARAM2:=VALUE2 ... 
```

The parameters are
- `voxel_size`: the size of the sparse voxels, the smaller the slower but it needs to be big enough to capture the geometry of the scene (default=`0.02`) 
- `max_range`: max range to crop the incoming point clouds (default=`1.5`)
- `max_freq`: this parameter allows to skip some incoming frames to match the specified maximum frequency (default=`3`)
- `pc_topic`: name of the PointCloud2 topic to used for odometry (default=`/camera/rgb/points`)


Then you will have to play your data from your sensor live or via a rosbag.
Please note that the implementation is optimised neither for performance nor robustness.
The performance will be degraded if the data is played too fast, if too many frames are skipped (parameter `max_freq` too low), or if the voxel size is too big (parameter `voxel_size`).
In the future, scripts for offline processing will address some of these issues at the cost of longer processing times.
However, the default parameters allow us to run the teddy bear dataset (c.f. next section) in real-time on a 7-ish-year-old Intel i5 + Nvidia 1050 GTX computer.
The first scan process will be slower as pykeops needs to compile some code. Subsequent uses will be faster.


### Examples

We have validated qualitatively this package using three datasets that are:
- the [teddy bear dataset](https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_teddy-2hz-with-pointclouds.bag) from _Sturm et al.(A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS2012)_. Based on the 2Hz version, runs real-time using the 50 first seconds only due to poor geometric info around timestamp 52sec: 
```
roslaunch gp_odometry gp_odometry.launch
rosbag play BAG_PATH -u 50
```
- the [cow and lady dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=iros2017/) from _Oleynikova et al.(Voxblox: Building 3D Signed Distance Fields for Planning, IROS2017)_. This dataset is challenging because of its larger scale, fairly aggressive motion, and the noise of the sensor (at longer range). The present package does perform well locally but drift occurs eventually. Need to use a small chunk of the dataset at a low playing rate:
```
roslaunch gp_odometry gp_odometry.launch voxel_size:=0.05 max_range:=4 max_freq:=5 pc_topic:=/camera/depth_registered/points
rosbag play BAG_PATH -s 10 -u 30 -r 0.1
```
- a [short handheld dataset](https://drive.google.com/file/d/1A7T39yyUxzrUiflUjkjPx6K-JTOe_sPv/view?usp=drive_link) using an Intel Realsense D455. Runs well in real-time on 1070 GTX.
```
roslaunch gp_odometry gp_odometry.launch pc_topic:=/camera/depth/color/points
rosbag play BAG_PATH
```

To play a rosbag at a slower rate (e.g. twice as slow), use the option `-r 0.5` (please adapt the value to your computational power).



### Citing

If you are using this code please cite our work:
```bibtex
@article{legentil2023accurate,
  title={Accurate Gaussian Process Distance Fields with applications to Echolocation and Mapping},
  author={{Le Gentil}, Cedric and Ouabi, Othmane-Latif and Wu, Lan and Pradalier, Cedric and Vidal-Calleja, Teresa},
  journal={arXiv preprint arXiv:2302.13005},
  year={2023}
}
```
and
```bibtex
@article{wu2023log,
  title={Log-GPIS-MOP: A Unified Representation for Mapping, Odometry, and Planning},
  author={Wu, Lan and Lee, {Ki Myung Brian} and {Le Gentil}, Cedric and Vidal-Calleja, Teresa},
  journal={IEEE Transactions on Robotics},
  year={2023},
  publisher={IEEE}
}
```
