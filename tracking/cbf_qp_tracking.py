import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import subprocess
from safe_control.tracking import LocalTrackingController

"""
Created on Feb 8, 2024
@author: Taekyung Kim

@description: This code implements a CBF-QP controller that tracks a set of waypoints.
It provides two dynamics models: Unicycle2D and DynamicUnicycle2D.
It has useful tools to analyze the union of sensing footprints and the safety of the robot.
This file has been refactored to use the safe_control library.

@required-scripts: safe_control submodule
"""

class UnicyclePathFollower:
    def __init__(self, type, X0, waypoints, dt=0.05, tf=100,
                  show_animation=False, plotting=None, env=None):
        self.type = type
        self.waypoints = waypoints
        self.dt = dt
        self.tf = tf
        
        self.plotting = plotting
        self.env = env
        
        # Map parameters to safe_control robot_spec
        self.robot_spec = {}
        if self.type == 'Unicycle2D':
            self.robot_spec['model'] = 'Unicycle2D'
            self.robot_spec['v_max'] = 1.0
            self.robot_spec['w_max'] = 0.5
            self.robot_spec['radius'] = 0.25
        elif self.type == 'DynamicUnicycle2D':
            self.robot_spec['model'] = 'DynamicUnicycle2D'
            self.robot_spec['a_max'] = 0.5
            self.robot_spec['w_max'] = 0.5
            self.robot_spec['v_max'] = 1.0 
            self.robot_spec['radius'] = 0.25
            
        # Initialize plotting if needed
        self.fig = None
        self.ax = None
        if show_animation:
            if self.plotting is None:
                self.fig = plt.figure()
                self.ax = plt.axes()
            else:
                self.ax, self.fig = self.plotting.plot_grid("Path Following")
        else:
            self.ax = plt.axes() # dummy
            
        # Controller configuration
        controller_type = {
            'pos': 'cbf_qp',
            'att': 'velocity_tracking_yaw' 
        }
        
        # Ensure X0 has correct dimension for the model
        if self.robot_spec['model'] == 'Unicycle2D':
            if X0.size == 2:
                # [x, y] -> [x, y, 0]
                X0 = np.pad(X0, (0, 1), 'constant')
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            if X0.size == 2:
                # [x, y] -> [x, y, 0, 0]
                X0 = np.pad(X0, (0, 2), 'constant')
            elif X0.size == 3:
                # [x, y, theta] -> [x, y, theta, 0]
                X0 = np.pad(X0, (0, 1), 'constant')

        self.controller = LocalTrackingController(
            X0=X0,
            robot_spec=self.robot_spec,
            controller_type=controller_type,
            dt=self.dt,
            show_animation=show_animation,
            save_animation=False, 
            ax=self.ax,
            fig=self.fig,
            env=self.env
        )
        
        # Safe Control expects obstacles to have specific format (7 elements), but utils.env provides [x,y,r]
        if self.controller.obs is not None and self.controller.obs.size > 0:
            if self.controller.obs.ndim == 1:
                self.controller.obs = self.controller.obs.reshape(1, -1)
            
            if self.controller.obs.shape[1] == 3:
                 # Pad to 7 columns: [x, y, r, 0, 0, 0, 0]
                 padding = np.zeros((self.controller.obs.shape[0], 4))
                 self.controller.obs = np.hstack((self.controller.obs, padding))
        
        
        # Set waypoints in the controller
        self.controller.set_waypoints(waypoints)
        
        # Additional state for compatibility
        self.unknown_obs = None
        
        # Expose robot for external access
        self.robot = self.controller.robot
        # Set test_type on robot if needed for visualization logic in original code
        self.robot.test_type = 'cbf_qp' 

    def set_unknown_obs(self, unknown_obs):
        self.unknown_obs = unknown_obs
        self.controller.set_unknown_obs(unknown_obs)
        self.robot.test_type = 'cbf_qp'


    def run(self, save_animation=False):
        print("===================================")
        print("============  CBF-QP  =============")
        print("Start following the generated path.")
        early_violation = 0
        unexpected_beh = 0

        self.controller.save_animation = save_animation
        if save_animation:
            self.controller.setup_animation_saving()

        max_steps = int(self.tf / self.dt)
        
        for i in range(max_steps):
            # Control step
            ret = self.controller.control_step()
            
            if ret == -2:
                print("ERROR in QP or Collision")
                unexpected_beh = -1
                break
            
            # Draw plot
            self.controller.draw_plot(pause=0.01)
            
            # Logic for compatibility:
            beyond_flag = 1 if ret == 1 else 0
            
            if self.controller.current_goal_index > 5:
                if i < int(5.0 / self.dt):
                     early_violation += beyond_flag
                unexpected_beh += beyond_flag 
                
                if beyond_flag and self.controller.show_animation:
                    print("Cumulative unexpected behavior: {}".format(unexpected_beh))

            if ret == -1: # All waypoints reached
                print("All waypoints reached.")
                break

        if save_animation:
             self.controller.export_video()

        print("=====   Simulation finished  =====")
        print("===================================\n")
        
        return unexpected_beh, early_violation

if __name__ == "__main__":
    dt = 0.05
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from utils import plotting
    from utils import env

    env_type = env.type
    if env_type == 1:
        tf = 100
    elif env_type == 2:
        tf = 30
        
    try:
        if env_type == 1:
            path_to_continuous_waypoints = os.getcwd()+"/output/240312-2128_large_env/state_traj.npy"
        elif env_type == 2:
            path_to_continuous_waypoints = os.getcwd()+"/output/240225-0430/state_traj.npy"
        
        waypoints = np.load(path_to_continuous_waypoints, allow_pickle=True)
        waypoints = np.array(waypoints, dtype=np.float64)
    except FileNotFoundError:
        print("Waypoint file not found, generating dummy waypoints for testing.")
        waypoints = np.array([[0,0], [1,1], [2,2], [3,3]], dtype=np.float64)
        path_to_continuous_waypoints = None

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    env_handler = env.Env()

    #type = 'Unicycle2D'
    type = 'DynamicUnicycle2D'
    path_follower = UnicyclePathFollower(type, x_init, waypoints, dt, tf, 
                                         show_animation=True,
                                         plotting=plot_handler,
                                         env=env_handler)

    if env_type == 1:
        unknown_obs = np.array([[13.0, 10.0, 0.5],
                                [12.0, 13.0, 0.5],
                                [15.0, 20.0, 0.5],
                                [20.5, 20.5, 0.5],
                                [24.0, 15.0, 0.5]])
    elif env_type == 2: 
        unknown_obs = np.array([[9.0, 8.8, 0.3]]) 

    path_follower.set_unknown_obs(unknown_obs)
    unexpected_beh, early_violation = path_follower.run(save_animation=False)
