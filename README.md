# VLM-Based Nonlinear Control for Precise Mobile Robot Inspection and Navigation

## Overview

This project presents an intelligent vision-based navigation and inspection framework for a mobile robot. The system integrates Vision-Language Model (VLM) based scene understanding, RGB-D geometric reconstruction, obstacle-aware waypoint planning, and Nonlinear Model Predictive Control (NMPC).

The proposed pipeline converts visual scene information into metric spatial information and uses closed-loop nonlinear control to autonomously navigate the robot to a safe inspection position while avoiding obstacles.

## Objective

The objective is to develop an autonomous mobile robot system capable of:

* Understanding a target inspection scene using visual perception.
* Identifying the target object and relevant obstacles.
* Estimating target and obstacle positions using RGB-D data.
* Generating a suitable inspection pose.
* Planning safe intermediate waypoints around obstacles.
* Controlling robot motion using NMPC.
* Reaching the target while maintaining obstacle clearance.
* Quantitatively evaluating navigation accuracy, safety, and computational performance.

## Key Features

* VLM-based semantic scene understanding
* RGB-D depth-based 3-D reconstruction
* Camera-to-robot coordinate transformation
* Automatic inspection pose generation
* Metric rectangular obstacle representation
* Obstacle-aware safe waypoint planning
* Nonlinear Model Predictive Control
* Receding-horizon closed-loop control
* Collision avoidance
* Navigation performance evaluation
* Trajectory and control visualization

## System Architecture

```text
RGB Image + Depth Image
          │
          ▼
   Scene Perception
     (VLM / Synthetic)
          │
          ▼
 Target + Obstacles
          │
          ▼
   RGB-D Processing
          │
          ▼
    3-D Estimation
          │
          ▼
 Camera → Robot Frame
          │
          ▼
   Inspection Pose
       Generation
          │
          ▼
 Obstacle Map + Safe
  Waypoint Planning
          │
          ▼
        NMPC
          │
          ▼
 Linear & Angular Velocity
       (v, ω)
          │
          ▼
    Robot State Update
          │
          └──────────────┐
                         │
                         ▼
                  Closed-Loop
                    Feedback
```

## Methodology

### 1. Scene Understanding

The input RGB image is analyzed to identify the inspection target and relevant obstacles. The system can operate using either:

* A local Vision-Language Model
* The synthetic ground-truth perception interface used for the benchmark experiment

The perception output contains object labels, bounding boxes, and navigation-related information.

### 2. RGB-D 3-D Reconstruction

The detected bounding boxes are combined with depth information to estimate metric 3-D positions.

Using camera intrinsics:

```text
X = (u - cx)Z / fx
Y = (v - cy)Z / fy
Z = depth
```

Robust depth estimation is performed using valid depth samples and median filtering.

### 3. Coordinate Transformation

The camera coordinate system is transformed into the robot coordinate system using:

```text
robot_x = camera_Z
robot_y = -camera_X
```

This produces target and obstacle locations in the robot's navigation frame.

### 4. Inspection Pose Generation

Instead of driving directly to the target, the system generates a pose at a preferred inspection distance.

The robot heading is calculated so that the robot faces the target from the generated inspection position.

### 5. Obstacle Representation

Detected obstacles are converted into metric rectangular regions in the robot coordinate frame.

A safety margin is applied to the obstacle geometry before navigation.

### 6. Safe Waypoint Planning

If the direct path to the inspection pose is blocked, intermediate waypoints are generated around the obstacle.

For the experimental scene, the successful navigation used:

```text
Start
  ↓
WP1 = [1.10, 0.10]
  ↓
WP2 = [4.40, 0.10]
  ↓
WP3 = [4.219, 3.375]
  ↓
Inspection Pose
```

### 7. Nonlinear Model Predictive Control

The robot is modeled using a nonlinear unicycle model:

```text
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = ω
```

The NMPC controller optimizes future control inputs over a finite prediction horizon.

The controller considers:

* Position error
* Heading error
* Obstacle clearance
* Control effort
* Terminal position accuracy
* Terminal heading accuracy
* Progress toward the goal

Only the first optimized control input is applied before the robot state is updated and the optimization is repeated.

## Experimental Configuration

| Parameter                   |     Value |
| --------------------------- | --------: |
| Control timestep            |     0.1 s |
| NMPC prediction horizon     |        30 |
| Prediction horizon duration |     3.0 s |
| Maximum linear velocity     |   1.0 m/s |
| Maximum angular velocity    | 1.0 rad/s |
| Safety distance             |     0.5 m |
| Obstacle influence distance |     2.0 m |
| Position tolerance          |    0.12 m |
| Heading tolerance           |        8° |

## Experimental Scene

The benchmark scene contains:

* **Target:** Chair
* **Obstacle:** Table

The detected target is represented by the bounding box:

```text
[85, 145, 155, 270]
```

The table is represented by:

```text
[70, 150, 236, 376]
```

The estimated target center in the camera coordinate system is:

```text
[-4.00, -0.66, 5.00] m
```

After camera-to-robot transformation:

```text
[5.00, 4.00] m
```

The generated inspection pose is approximately:

```text
[4.219, 3.375, 0.675]
```

## Results

The final successful experiment achieved:

| Metric                        |      Result |
| ----------------------------- | ----------: |
| Inspection goal reached       |    **True** |
| Waypoint navigation completed |    **True** |
| Collision count               |       **0** |
| Minimum obstacle clearance    | **1.179 m** |
| Final position error          |     0.696 m |
| Final heading error           |      0.061° |
| Path length                   |     8.933 m |
| Maximum linear velocity       |     1.0 m/s |
| Maximum angular velocity      |   1.0 rad/s |
| Average NMPC computation time |   306.38 ms |

The navigation remained collision-free while successfully completing the planned waypoint sequence and reaching the inspection state according to the implemented goal criteria.

## Result Visualizations

The generated results include:

* `inspection_trajectory.png` — robot trajectory and obstacle map
* `linear_velocity.png` — linear velocity profile
* `angular_velocity.png` — angular velocity profile
* `clearance.png` — obstacle clearance during navigation

## Project Structure

```text
vlm_nmpc_complete/
│
├── main_pipeline.py
├── navigation.py
├── vlm_scene.py
├── rgbd_processor.py
├── generate_figure7.py
│
├── data/
│   └── scene_01/
│       ├── rgb.png
│       ├── depth.png
│       └── ground_truth.json
│
├── results/
│   ├── scene_result.json
│   ├── metrics.csv
│   ├── trajectory.csv
│   ├── controls.csv
│   ├── inspection_trajectory.png
│   ├── linear_velocity.png
│   ├── angular_velocity.png
│   └── clearance.png
│
├── figures/
│   └── methodology figures
│
├── report/
│   └── report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository and create a Python virtual environment.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd vlm_nmpc_complete
```

Create and activate the environment:

### Windows

```powershell
python -m venv tg_venv
tg_venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Pipeline

The default synthetic benchmark can be executed using:

```powershell
python main_pipeline.py
```

The default configuration uses the scene files located in:

```text
data/scene_01/
```

The generated results are saved automatically to:

```text
results/
```

## VLM Mode

The pipeline also supports local VLM-based perception.

Example:

```powershell
python main_pipeline.py --perception vlm
```

The VLM model can be specified using:

```powershell
python main_pipeline.py --perception vlm --model HuggingFaceTB/SmolVLM-256M-Instruct
```

The synthetic mode is provided as a reproducible benchmark interface for evaluating the downstream RGB-D, planning, and NMPC components independently of VLM inference variability.

## Applications

The proposed framework can be extended to:

* Autonomous robotic inspection
* Industrial equipment inspection
* Indoor mobile robot navigation
* Infrastructure monitoring
* Warehouse robotics
* Service robotics
* Target-based autonomous navigation
* Vision-guided robotic inspection
* Human-robot environments

## Limitations

The current implementation is primarily evaluated in a synthetic benchmark environment. The robot dynamics are represented using a unicycle model, and the obstacle geometry is simplified to rectangular regions.

The current evaluation also does not include real-world actuator dynamics, localization uncertainty, moving obstacles, or physical robot experiments.

## Future Work

Future improvements can include:

* Real-world mobile robot deployment
* Real-time VLM perception
* Dynamic obstacle handling
* SLAM-based mapping
* Uncertainty-aware RGB-D estimation
* More realistic robot dynamics
* Hardware-in-the-loop NMPC
* Multi-object inspection
* Adaptive inspection distance
* Real-time GPU-accelerated NMPC

## References

The theoretical foundations of the project are based on research in vision-language models, robotics, obstacle avoidance, numerical optimization, and model predictive control.

See the project report for the complete IEEE-formatted reference list.

## Author

**Akhila Rao**

Project: **VLM-Based Nonlinear Control for Precise Mobile Robot Inspection and Navigation**
