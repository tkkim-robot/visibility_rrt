# visibility_rrt

This repository contains the implementation of the Visibility-Aware RRT* algorithm, a sampling-based path planning method that generates safe and efficient global reference paths for robots with limited sensing capabilities in partially unknown environments. The algorithm incorporates a collision avoidance Control Barrier Function (CBF) and a novel visibility CBF to ensure that the generated paths are not only collision-free but also respect the robot's limited perception capabilities. Please see our paper ["Visibility-Aware RRT* for Safety-Critical Navigation of Perception-Limited Robots in Unknown Environments"](https://ieeexplore.ieee.org/document/10930526) for more details.


<div align="center">
<img src="https://github.com/user-attachments/assets/7d889ef0-ca4e-4788-9b11-7d37fea916a6" width="700px">

<div align="center">

[[Homepage]](https://www.taekyung.me/visibility-rrt)
[[Arxiv]](https://arxiv.org/abs/2406.07728)
[[Paper]](https://ieeexplore.ieee.org/document/10930526)
[[Video]](https://youtu.be/l4vlMTf8s74)
[[Research Group]](https://dasc-lab.github.io/)

</div>
</div>

# Installation

This repositry only requires common libraries: `numpy`, `matplotlib`, and `scipy`. 

For running CBF-QP, it requires `cvxpy`, `gurobipy`, and `shapely`.

# How to Run Example

## Path Planning
You can run our test example by:

```bash
python visibility_rrtStar.py
```

Alternatively, you can import `VisibilityRRTStar` from 'visibility_rrtStar.py'.

```python
from visibility_rrtStar import VisibilityRRTStar
rrt_star = VisibilityRRTStar(x_start=x_start, x_goal=x_goal,
                              visibility=True,
                              collision_cbf=True)

# assuming you have set the workspace (environment) in rrt_star.env
waypoints, _ , _ = rrt_star.planning()
```

You can test the baseline algorithms:
- [LQR-RRT*](https://ieeexplore.ieee.org/document/6225177):
    - by setting `visibility=False` and `collision_cbf=False`.
- [LQR-CBF-RRT*](https://arxiv.org/abs/2304.00790): 
    - by setting `visibility=False` and `collision_cbf=True`.

The sample results of the generated global paths:

|                                                     Visibility-Aware RRT* (w/ visibility CBF)                                                    |                                                                        LQR-CBF-RRT* (w/o visibility CBF)                |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/6e6976d1-09b2-4189-b186-e4b59ece3efb"  height="200px"> | <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/2831ad29-88d9-4b36-a7e3-87566a06bad3"  height="200px"> |


## Path Tracking (CBF-QP)
Then, you can test a CBF-QP controller to track the resulted path by specifying the saved path:

```bash
python tracking/cbf_qp_tracking.py
```

Alternatively, you can import `UnicyclePathFollower` from 'tracking/cbf_qp_tracking.py'.

```python
from tracking.cbf_qp_tracking import UnicyclePathFollower

x_init = waypoints[0]
path_follower = UnicyclePathFollower('DynamicUnicycle2D', x_init, 
                                         waypoints)
_ = path_follower.run()
```

You can also set hidden obstacles, which were not considered during the path planning phase:
```python
unknown_obs = np.array([[x_center, y_center, radius]]) 
path_follower.set_unknown_obs(unknown_obs)
```
The hidden obstacles are depicted in orange circle, and can be detected by the onboard sensor of the robot. The detection points will be depicted in red points.

You can test with two dynamics model:
- `'Unicycle2D'`: A standard unicycle model with translational velocity and rotational speed as control input.
- `'DynamicUnicycle2D'`: It uses translational acceleration as control input and treats translational velocity as a state.

The sample results of the CBF-QP tracking (FOV: 45°):

|                                                     Tracking a Path of the Visibility-Aware RRT*                                                    |                                                                       Tracking a Path of the LQR-CBF-RRT*                 |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/6c3fc15f-1796-49b8-8892-79b1aae2598a"  height="200px"> | <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/0ed47755-5ce4-4233-9a22-b3cba408016e"  height="200px"> |


While the CBF-QP tracking a path from the baseline algorithm (which is agnostic to the sensing capability), the robot detects the hidden obstacle too late, leaving no feasible solution for the CBF-QP to avoid the obstacle.

## Path Tracking (GateKeeper)
You can also simulate a GateKeeper controller. The red shaded area in front of the robot depicts the minimum breaking distance at the current speed. If this area lies outside of the sensed collision-free space, the next waypoint to follow (the nominal trajectory) is deemed unsafe. 

<p align="center">
    <img width="300" alt="env2_gatekeeper_baseline" src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/f98224f7-42b7-4798-8360-4e15bd345454">
</p>

In the code, the same `path_follower` instance returns the number of such violations:
```python
unexpected_beh = path_follower.run()
```
If the output is larger than 0, it means that the nominal trajectory will be rejected by the GateKeeper algorithm more than one time.

The sample results of the GateKeeper tracking (FOV: 70°):

|                                                     Tracking a Path of the Visibility-Aware RRT*                                                    |                                                                       Tracking a Path of the LQR-CBF-RRT*                 |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/57007625-e8f8-4b13-b2c5-2d42a1f59f9e"  height="200px"> | <img src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/f98c6be0-1e18-461c-bba4-cb1b1e18b328"  height="200px"> |


## LQR-CBF-Steer
You can also visualize how the LQR-CBF-Steer function works in the algorithm. An example of the steering process with the visibility CBF can be visualized by runnning:

```bash
python LQR_CBF_planning.py
```

<p align="center">
    <img width="350" alt="lqr_cbf_steer" src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/0fdc0a5b-d465-46ee-90c4-f5b1b6f67175">
</p>

## Efficiency Comparison
The figure illustrates the average number of vertices maintained in the tree at iterations 1000, 2000, and 3000 over 100 runs. The results show that the Visibility-Aware RRT* consistently maintains fewer vertices compared to the two baseline algorithms, demonstrating its efficiency. By maintaining fewer nodes, it can reduce the computational complexity of the `choose_parent` and `rewire` functions. The additional vertices in the compared baselines arise from their lack of consideration for input constraints and visibility constraint, respectively.

<p align="center">
    <img width="350" alt="vertices" src="https://github.com/tkkim-robot/visibility-rrt/assets/40379815/38ab90d5-f967-49d7-9d4a-0502895f273a">
</p>

# Parameters Description
These are the important parameters to tune:

### `iter_max`

- The maximum number of iterations allowed for the RRT* algorithm. A higher value allows for more exploration and potentially better paths but increases the runtime.
- We recommend 1,000 - 3,000, depending on the size of your environment.

### `max_sampled_node_dist`
- The maximum distance between the sampled node and the nearest node in the tree. This parameter controls the step size of the tree expansion.
- A smaller value results in a denser tree but may require more iterations to reach the goal.

### `rewiring_radius`

- The radius used to find nearby nodes for rewiring. 

### `fov_angle` and `cam_range`

- For planner, they are in 'visibility_cbf.py'.
- For controller, they are in 'tracking/robot.py'.
- They encode the sensing capability of the onboard sensor.

See our paper for further information.

# Citing

If you find this repository useful, please consider citing our paper:

```
@article{kim2025visibility-aware, 
    author    = {Kim, Taekyung and Panagou, Dimitra},
    title     = {{Visibility-Aware RRT* for Safety-Critical Navigation of Perception-Limited Robots in Unknown Environments}},
    journal = {IEEE Robotics and Automation Letters},
    shorttitle = {{Visibility}-{RRT}*},
    year      = {2025},
    volume = {10},
    number = {5},
    pages = {4508--4515},
    doi = {10.48550/arXiv.2406.07728},
}

```

# Related Works

This repository was built based on the [implementation of LQR_CBF_rrtStar](https://github.com/mingyucai/LQR_CBF_rrtStar), that I had contributed before. Thanks for the great work of [gy2256](https://github.com/gy2256) and [mingyucai](https://github.com/mingyucai).
