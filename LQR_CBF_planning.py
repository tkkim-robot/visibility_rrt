
import math
import scipy
import time
import matplotlib.pyplot as plt
import numpy as np

from cbf import CBF
from visibility_cbf import Visibility_CBF
import utils.env as env
from utils.node import Node
from utils.utils import angular_diff, angle_normalize, calculate_fov_points

"""

Created on Jan 22, 2024
@author: Taekyung Kim
@modified-by: Ihab S. Mohamed (Indiana University -- Bloomington) - Jan 2026

@description: This code implements a discrete-time finite horizon LQR with unicycle model using velocity control. 
This code also provides a path planning function with the computed LQR gain, both using QP and not using QP.
The planning without using QP is the LQR-CBF-Steer algorithm in the paper.
They are similar to "motion_plannig_with/without_QP()" in cbf.py, but they use LQR gains instead.
The main function shows tracking a randomly generated goal points with LQR planning. (can turn on/off QP)

@note: This code is a refactorized version of the LQR-CBF-RRT* paper with non-linear dynamics.
The code baseline is from Guang Yang.
Please see this origina code for detail : https://github.com/mingyucai/LQR_CBF_rrtStar/blob/main/nonlinear_dynamic_model/LQR_nonlinear_planning.py
Currently. only supports unicycle model with velocity control.

@required-scripts: cbf.py, visibility_cbf.py, env.py

@being-used-in: LQR_CBF_rrtStar.py

@update: Updated version addressing five critical issues:
    1. Fixed LQR gain calculation to use P[i+1] instead of P[i] (Principle of Optimality)
    2. Fixed Jacobian linearization error: A[1,2] now uses v*cos(theta) instead of v*sin(theta)
    3. Fixed linearization at zero velocity: now uses non-zero reference velocity (1.0) to maintain controllability
    4. Fixed affine linearization offset: f_xu third element now uses yaw (control input) instead of theta (state)
    5. Changed state propagation to use nonlinear dynamics instead of linearized model for better accuracy
    
"""

class LQR_CBF_Planner:

    def __init__(self, visibility=False, collision_cbf=False):

        self.N = 3  # number of state variables
        self.M = 2  # number of control variables
        self.DT = 0.1  # discretization step

        self.MAX_TIME = 4.0  # Maximum simulation time
        self.GOAL_DIST = 0.6 # m

        # LQR gain is invariant
    
        self.Q = np.matrix("25 0 0; 0 25 0; 0 0 100")  
        self.R = np.matrix("5.0 0; 0 0.05")  

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        # TODO: currently not supporting rectangle and boundary obstacle
        # self.obs_rectangle = self.env.obs_rectangle
        # self.obs_boundary = self.env.obs_boundary
        self.collision_cbf = CBF(self.obs_circle)
        self.visibility_cbf = Visibility_CBF()

        self.visibility_cbf_flag= visibility
        self.collision_cbf_flag = collision_cbf
        self.cx = 0.0 # Critical point
        self.cy = 0.0 

    def compute_critical_point(self, rx, ry, ryaw, gx, gy, gtheta):

        cam_range = self.visibility_cbf.cam_range
        fov_angle = self.visibility_cbf.fov

        # Compute the relative angle between the goal node and the robot's heading
        delta_theta = angle_normalize(gtheta - ryaw)
        # Check if the relative angle lies outside of the current FOV
        if abs(delta_theta) > fov_angle / 2:
            # Compute the tube radius
            tube_radius = cam_range * math.sin(fov_angle / 2)

            # Compute the slope and y-intercept of the line connecting the robot and goal node
            if gx - rx != 0:
                slope = (gy - ry) / (gx - rx)
                y_intercept = ry - slope * rx
            else:
                slope = float('inf')
                y_intercept = float('inf')

            # Compute the slope and y-intercept of the tube boundary lines
            tube_slope = math.tan(ryaw)
            if delta_theta > 0:
                # Goal node is on the right side of the robot's heading angle
                tube_y_intercept = ry - tube_slope * rx + tube_radius / math.cos(ryaw)
            else:
                # Goal node is on the left side of the robot's heading angle
                tube_y_intercept = ry - tube_slope * rx - tube_radius / math.cos(ryaw)

            cx = (tube_y_intercept - y_intercept) / (slope - tube_slope)
            cy = slope * cx + y_intercept

        # the critical point is in the FOV, but might or might not be in the current sensing range
        else:
            MAX_DIST_CRITICAL = math.cos(math.pi/2 - fov_angle/2) * cam_range # 3 is range of the sensor
            dist_to_critical = math.hypot(gx-rx, gy-ry)
            dist_to_critical = min(dist_to_critical, MAX_DIST_CRITICAL)
            cx = rx + dist_to_critical * math.cos(gtheta)
            cy = ry + dist_to_critical * math.sin(gtheta)

        return cx, cy

    
    def lqr_cbf_planning(self, start_node, goal_node, LQR_gain, solve_QP = False, show_animation = True):

        sx = start_node.x
        sy = start_node.y
        if start_node.yaw is None:
            raise RuntimeError("start node' yaw is not specified")
        else:
            stheta = start_node.yaw
        stheta = angle_normalize(stheta)

        gx = goal_node.x
        gy = goal_node.y
        if goal_node.yaw is None:
            gtheta = math.atan2(gy-sy, gx-sx)
        else:
            gtheta = goal_node.yaw
        gtheta = angle_normalize(gtheta)

        # TODO: this does not necessary when only using QPConstraint
        self.collision_cbf.set_initial_state(np.array([sx, sy, stheta]))
        self.visibility_cbf.set_initial_state(np.array([sx, sy, stheta]))


        # Linearize system model
        # FIX 3: Use non-zero reference velocity to maintain controllability
        # Using v=1.0 preserves coupling between heading and position in the Jacobian
        xd = np.matrix([[gx], [gy], [gtheta]])
        ud = np.matrix([[1.0], [0]])  # Changed from [[0], [0]]
        self.A, self.B, self.C = self.get_linear_model(xd, ud)

        # Check the hash table to store LQR feedback Gain
        waypoint = (gx, gy)
        if waypoint in LQR_gain:
            # print('found one prebious gain')
            self.K = LQR_gain[waypoint]
        else:
            self.K = self.finite_dLQR(self.A, self.B, self.Q, self.R)
            LQR_gain[waypoint] = self.K

        # initialize robot trajectroy, start from current state
        rx, ry, ryaw = [sx], [sy], [stheta]
        error = []

        xk = np.array([sx, sy, stheta]).reshape(3, 1)  # State vector

        found_path = False

        if show_animation:
            fov_lines = []
            fov_fills = []

        i = 0 # idx
        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            #MAX_DIST_CRITICAL = 5.0 # [m]
            cam_range = self.visibility_cbf.cam_range

            cx, cy = self.compute_critical_point(rx[-1], ry[-1], ryaw[-1], gx, gy, gtheta)
            self.cx = cx
            self.cy = cy

            self.visibility_cbf.set_critical_point(np.array([cx, cy]))

            x = xk - xd
            x[2, 0] = angular_diff(xk[2, 0], xd[2, 0])
            #x = xd - xk
            u = self.K[i] @ x
            # print("x ", x)
            # print("u ", u)
            # print("K ", self.K[i])
            i += 1
            
            # animation
            if show_animation: 
                # for stopping simulation with the 'esc' key
                # Remove previous FOV and triangle
                for line in fov_lines:
                    line.remove()
                fov_lines.clear()
                for fill in fov_fills:
                    fill.remove()
                fov_fills.clear()

                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(cx, cy, "om")
                plt.plot(rx, ry, "-r")

                robot_position = (rx[-1], ry[-1])
                yaw = ryaw[-1]

                robot_circle = plt.Circle(robot_position, 0.1, color='blue', fill=True)
                plt.gca().add_patch(robot_circle)

                # Draw the yaw line
                yaw_line_end = (robot_position[0] + math.cos(yaw), robot_position[1] + math.sin(yaw))
                plt.plot([robot_position[0], yaw_line_end[0]], [robot_position[1], yaw_line_end[1]], 'g-')

                # Calculate and draw the FOV
                fov_left, fov_right = calculate_fov_points(robot_position, yaw, fov_angle=self.visibility_cbf.fov, cam_range=cam_range)
                fov_lines.append(plt.plot([robot_position[0], fov_left[0]], [robot_position[1], fov_left[1]], 'k-')[0])
                fov_lines.append(plt.plot([robot_position[0], fov_right[0]], [robot_position[1], fov_right[1]], 'k-')[0])
                fov_lines.append(plt.plot([fov_left[0], fov_right[0]], [fov_left[1], fov_right[1]], 'k-')[0])

                # Calculate FOV points at the start position 
                fov_left_init, fov_right_init = calculate_fov_points((rx[0], ry[0]), gtheta, fov_angle=self.visibility_cbf.fov, cam_range=cam_range)

                # Calculate FOV points at the current position
                current_fov_left, current_fov_right = calculate_fov_points(robot_position, gtheta, fov_angle=self.visibility_cbf.fov, cam_range=cam_range)

                # Draw dashed lines for the FOV boundaries
                fov_lines.append(plt.plot([fov_left_init[0], current_fov_left[0]], [fov_left_init[1], current_fov_left[1]], 'k--', alpha=0.5)[0])
                fov_lines.append(plt.plot([fov_right_init[0], current_fov_right[0]], [fov_right_init[1], current_fov_right[1]], 'k--', alpha=0.5)[0])

                # Fill the FOV tube
                fov_fills.append(plt.fill([fov_left_init[0], current_fov_left[0], current_fov_right[0], fov_right_init[0]],
                        [fov_left_init[1], current_fov_left[1], current_fov_right[1], fov_right_init[1]], 'grey', alpha=0.1)[0])

                # Fill the FOV triangle
                fov_fills.append(plt.fill([robot_position[0], fov_left[0], fov_right[0]], [robot_position[1], fov_left[1], fov_right[1]], 'k', alpha=0.1)[0])

                plt.axis("equal")
                plt.title("iteration: {}".format(i))
                plt.xlim(-10, 10)
                plt.ylim(-10, 10)
                plt.pause(0.5)

            if solve_QP:
                #solve QP with CBF, update control input u
                try:
                    u = np.array(u).squeeze() # convert matrix to array
                    u = self.collision_cbf.QP_controller([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u, model = "unicycle")
                    u = np.matrix(u).reshape(2, -1) # convert array to matrix
                except:
                    print('The CBF-QP at current steering step is infeasible')
                    break
            else:
                # check if LQR control input is safe with respect to CBF constraint, not solving QP
                if self.collision_cbf_flag:
                    collision_cbf_constraint = self.collision_cbf.QP_constraint([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u, model = "unicycle_velocity_control")
                    if not collision_cbf_constraint:
                        #print("violated collision cbf constraint")
                        break
                if self.visibility_cbf_flag:
                    visibility_cbf_constraint = self.visibility_cbf.QP_constraint([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u, model = "unicycle_velocity_control")
                    if not visibility_cbf_constraint:
                    #print("violated visibility cbf constraint")
                    # violate either of constraint
                        break

            # update current state
            # Linear form (original)
            # xk = self.A @ xk + self.B @ u 
            
            # FIX 5: Use nonlinear dynamics for state propagation instead of linearized model
            v = u[0, 0]
            yaw = u[1, 0]
            xk[0, 0] = xk[0, 0] + self.DT * v * np.cos(xk[2, 0])
            xk[1, 0] = xk[1, 0] + self.DT * v * np.sin(xk[2, 0])
            xk[2, 0] = xk[2, 0] + self.DT * yaw
            
            theta_k = angle_normalize(xk[2,0])
            xk[2,0] = theta_k

            # if collisino_cbf_flag is false, then do normal collision check
            if not self.collision_cbf_flag:
                collision = self.collision_cbf.collision_check([xk[0, 0], xk[1, 0], xk[2,0]], model = "unicycle_velocity_control")
                if collision:
                    # print(xk[0, 0], xk[1, 0])
                    # print("collision")
                    break
                

            rx.append(xk[0, 0])
            ry.append(xk[1, 0])
            ryaw.append(xk[2, 0])

            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break


        if show_animation:
            # Remove previous FOV and triangle
            for line in fov_lines:
                line.remove()
            fov_lines.clear()
            for fill in fov_fills:
                fill.remove()
            fov_fills.clear()

        if not found_path:
            #print("Cannot found !!")
            return [rx, ry, ryaw], error, found_path

        #print("Fonud path to goal")
        return [rx, ry, ryaw], error, found_path


    def finite_dLQR(self, A, B, Q, R):
        """
        Finite horizon discrete-time LQR
        
        FIX 1: Corrected to use P[i+1] for gain calculation (Principle of Optimality)
        The optimal gain at time i should be derived from the future cost-to-go P[i+1],
        not the current cost-to-go P[i].
        """
        N = int(self.MAX_TIME / self.DT)
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        Qf = Q
        # LQR via Dynamic Programming
        P[N] = Qf
        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # state cost matrix
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)   
            # print("P[{}]:\n{}".format(i-1, P[i-1]))
        K = []
        for i in range(0, N, 1):
            # FIX 1: Use P[i+1] instead of P[i] for correct dynamic programming
            K.append(-np.linalg.inv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A)
        
        return K

    def infinite_dLQR(self, A, B, Q, R):
        """
        * Currently not used.
        Solve the infinite horizon discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))

        #eigVals, eigVecs = scipy.linalg.eig(A-B*K)        
        return -K
    
    def get_linear_model(self, x_bar, u_bar):
        """
        Computes the Discrete-time LTI approximated state space model x' = Ax + Bu + C
        
        FIX 2: Corrected Jacobian linearization for y-position w.r.t. theta
        For unicycle model: dy/dt = v*sin(theta)
        Partial derivative: ∂(dy/dt)/∂theta = v*cos(theta), not v*sin(theta)
        """

        x = x_bar[0]
        y = x_bar[1]
        theta = x_bar[2]

        v = u_bar[0]
        yaw = u_bar[1]

        A = np.zeros((self.N, self.N))
        A[0, 2] = -v * np.sin(theta)
        A[1, 2] = v * np.cos(theta)  # FIX 2: Changed from v*sin(theta) to v*cos(theta)
        A_lin = np.eye(self.N) + self.DT * A

        B = np.zeros((self.N, self.M))
        B[0, 0] = np.cos(theta)
        B[1, 0] = np.sin(theta)
        B[2, 1] = 1
        B_lin = self.DT * B

        # FIX 4: Changed third element from theta (state) to yaw (control input)
        f_xu = np.array(
            [v * np.cos(theta), v * np.sin(theta), yaw]
        ).reshape(self.N, 1)
        C_lin = self.DT * (
            f_xu - np.dot(A, x_bar.reshape(self.N, 1)) - np.dot(B, u_bar.reshape(self.M, 1))
        )

        return np.round(A_lin, 4), np.round(B_lin, 4), np.round(C_lin, 4)



if __name__ == '__main__':

    print(__file__ + " start!!")

    import random

    SHOW_ANIMATION = True
    SOLVE_QP = False

    ntest = 5  # number of goal
    area = 10.0  # sampling area

    lqr_cbf_planner = LQR_CBF_Planner(visibility=False)

    # initialize a hash table for storing LQR gain
    # TODO: this should be modified to be optional 
    LQR_gain = dict()

    for i in range(ntest):
        start_time = time.time()

        gx = random.uniform(-area, area)
        gy = random.uniform(-area, area)

        sx = 0.0
        sy = 0.0
        stheta = math.atan2(gy-sy, gx-sx)

        # add a small noise to the stheta
        stheta += random.uniform(-math.pi/2, math.pi/2)
        #stheta -= math.radians(75/2) + 0.4 # for testing

        start_node = Node([sx, sy, stheta])
        goal_node = Node([gx, gy])

        print("goal", gy, gx)

        rtraj, error, foundpath = lqr_cbf_planner.lqr_cbf_planning(start_node, goal_node, LQR_gain=LQR_gain, solve_QP = SOLVE_QP, show_animation=SHOW_ANIMATION)
        rx, ry, ryaw = rtraj

        print("time of running LQR: ", time.time() - start_time)

        if not SHOW_ANIMATION:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.plot(sx, sy, "or")
            ax1.plot(gx, gy, "ob")
            ax1.plot(lqr_cbf_planner.cx, lqr_cbf_planner.cy, "om")
            print(lqr_cbf_planner.cx, lqr_cbf_planner.cy)
            ax1.plot(rx, ry, "-r")
            ax1.grid()
            
            ax2.plot(error, label="errors")
            ax2.legend(loc='upper right')
            ax2.grid()
            plt.show()

    print("end main")