
import matplotlib.pyplot as plt
import math
import numpy as np

from scipy.integrate import solve_ivp
from gurobipy import *

"""
Created on Jan 21, 2024
@author: Taekyung Kim

@description: This code implements a collision avoidance CBF with an unicycle model using velocity control.
This code also provides a path generation function with the CBF constraint.
The main function shows some examples.

@note: This code is a rewritten version of the CBF-RRT paper with
non-linear dynamics. The code baseline is from Guang Yang.
Currently. only supports unicycle model with velocity control.
The obstacle radius r includes the inflated radius, which is the radius of the robot + maximum tracking error of the local tracking controller.

@being-used-in: LQR_CBF_planning.py
"""


class CBF:
    def __init__(self, obstacle_list):
        self.DT = 0.1  # Integration Length
        self.N = 50  # Number of Control Updates
        # self.k = 6 # k nearest neighbor obstacles that will be used for generating CBF constraint
        self.cbf_constraints_sensing_radius = 5

        # TODO: should be tuned and add explanation
        self.k1_unicyle_cbf = 5  # CBF coefficient for h(x)
        self.k2_unicyle_cbf = 3  # CBF coefficient for Lfh(x)

        self.x_obstacle = obstacle_list
        self.w_lower_lim = -5  # only used in QP
        self.w_upper_lim = 5
        self.unicycle_constant_v = 1.0
        self.robot_radius = 0.3

    def set_initial_state(self, initial_state):
        self.init_state = np.array(initial_state)

    def QP_controller(self, x_current, u_ref, model="unicycle"):
        if model == "unicycle":
            # TODO: Make sure if we want Lyapunov function
            self.m = Model("CBF_QP_Unicycle")
            x = x_current[0]
            y = x_current[1]
            theta = x_current[2]

            # self.unicycle_constant_v # Set constant linear velcoity to avoid mixed relative degree control
            v = u_ref[0]
            yaw_rate = u_ref[1]

            # remove any existing constraints before
            self.m.remove(self.m.getConstrs())
            # Control angular velocity
            self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,
                                   vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")

            # Initialize Cost Function, minimize distanc to u_ref
            self.cost_func = (self.w-yaw_rate)*(self.w-yaw_rate)
            self.m.setObjective(self.cost_func, GRB.MINIMIZE)

            # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0
            for i in range(0, len(self.x_obstacle)):
                xo = self.x_obstacle[i][0]
                yo = self.x_obstacle[i][1]
                r = self.x_obstacle[i][2]

                h = (x-xo)**2+(y-yo)**2-r**2
                Lfh = 2*v*math.cos(theta)*(x-xo) + 2*v*math.sin(theta)*(y-yo)
                LfLfh = 2*(v**2)*math.cos(theta)**2 + \
                    2*(v**2)*math.sin(theta)**2
                LgLfh = 2*v*math.cos(theta)*(y-yo) - 2*v*math.sin(theta)*(x-xo)

                self.m.addConstr(LfLfh + LgLfh*self.w + self.k2_unicyle_cbf *
                                 Lfh + self.k1_unicyle_cbf*h >= 0, "CBF_constraint")

            # Stop optimizer from publsihing results to console - remove if desired
            self.m.Params.LogToConsole = 0

            # Solve the optimization problem
            self.m.optimize()
            self.solution = self.m.getVars()
            yaw_rate_qp = self.solution[0].x

            return np.array([v, yaw_rate_qp])

    def motion_planning_with_QP(self, u_ref, model="unicycle_velocity_control"):
        x_current = self.init_state
        self.x = np.zeros((0, 3))
        self.u = np.zeros((0, 2))

        # slight abuse of notation, delta_t is the time step for each control update during 0.2 s (= self.DT)
        delta_t = self.DT/self.N
        # number of DT for planning. In this example, we use u_ref (defined for 0.2 s = DT) repeatedly.
        time_step = 50

        for k in range(time_step):
            for i in range(0, self.N):
                self.x = np.vstack((self.x, x_current))
                u_current_ref = np.array([self.unicycle_constant_v, u_ref[k]])
                # qp_controller(x_i, u_k) where u_k is constant during self.N steps
                u_optimized = self.QP_controller(x_current, u_current_ref)
                self.u = np.vstack((self.u, u_optimized))

                # Update staet using unicycle kinematics
                theta = x_current[2]
                dx = u_optimized[0] * np.cos(theta) * delta_t
                dy = u_optimized[0] * np.sin(theta) * delta_t
                dtheta = u_optimized[1] * delta_t
                x_current = x_current + np.array([dx, dy, dtheta])

        return self.x, self.u

    def find_obstacles_within_cbf_sensing_range(self, x_current, x_obstacles):
        obstacles_idx = []

        for i in range(len(x_obstacles)):
            distance = math.hypot(
                x_current[0] - x_obstacles[i][0], x_current[1] - x_obstacles[i][1])
            if distance <= self.cbf_constraints_sensing_radius:
                obstacles_idx.append(i)

        return obstacles_idx

    def QP_constraint(self, x_current, u_ref, model="unicycle_velocity_control"):
        """
        return:
            True: if CBF constraint is satisfied
            False: if CBF constraint is violated
        """
        if model == "unicycle_velocity_control":
            x = x_current[0]
            y = x_current[1]
            theta = x_current[2]

            v = u_ref[0]  # Linear velocity
            w = u_ref[1]  # Angular velocity

            # States: x, y, theta
            obstacle_index = self.find_obstacles_within_cbf_sensing_range(
                x_current, self.x_obstacle)

            if obstacle_index:
                minCBF = float('inf')

                for index in obstacle_index:
                    xo = self.x_obstacle[index][0]
                    yo = self.x_obstacle[index][1]
                    r = self.x_obstacle[index][2] + self.robot_radius

                    # Unicycle with velocity control
                    h = (x-xo)**2+(y-yo)**2-r**2
                    Lfh = 2*v*math.cos(theta)*(x-xo) + 2 * \
                        v*math.sin(theta)*(y-yo)
                    LfLfh = 2*(v**2)*math.cos(theta)**2 + \
                        2*(v**2)*math.sin(theta)**2
                    LgLfh = 2*v*math.cos(theta)*(y-yo) - \
                        2*v*math.sin(theta)*(x-xo)

                    CBF_Constraint = LfLfh + LgLfh*w + self.k2_unicyle_cbf*Lfh + self.k1_unicyle_cbf*h

                    if CBF_Constraint < minCBF:
                        minCBF = CBF_Constraint

                    if minCBF < 0:
                        return False
        return True

    def collision_check(self, x_current, model="unicycle_velocity_control"):
        """
            Do normal collision check for RRT*
        return:
            True: collision
            False: no collision
        """
        if model == "unicycle_velocity_control":
            x = x_current[0]
            y = x_current[1]
            # theta = x_current[2]

            # States: x, y, theta
            obstacle_index = self.find_obstacles_within_cbf_sensing_range(
                x_current, self.x_obstacle)

            if obstacle_index:
                minCBF = float('inf')

                for index in obstacle_index:
                    xo = self.x_obstacle[index][0]
                    yo = self.x_obstacle[index][1]
                    r = self.x_obstacle[index][2]

                    # Unicycle with velocity control
                    h = (x-xo)**2+(y-yo)**2-r**2
                    if h < minCBF:
                        minCBF = h

                    if minCBF < 0:
                        return True
        return False

    def motion_planning_without_QP(self, u_ref, model="unicycle_velocity_control"):
        if model == "unicycle_velocity_control":
            x_current = self.init_state
            self.x = np.zeros((0, 3))  # generated state traj
            self.u = np.zeros((0, 2))  # generated control traj

            for i in range(len(u_ref)):
                u_current = np.array([self.unicycle_constant_v, u_ref[i]])
                if not self.QP_constraint(x_current, u_current, model="unicycle_velocity_control"):
                    # stop generating traj if CBF constraint is violated
                    break
                else:
                    def unicycle_model_velocity_control(t, y):
                        return [math.cos(y[2]) * self.unicycle_constant_v, math.sin(y[2]) * self.unicycle_constant_v, u_ref[i]]

                    solution = solve_ivp(fun=unicycle_model_velocity_control, t_span=[
                                         0, self.DT], y0=x_current, dense_output=True)
                    # y[:, -1] to obtain the state after integration, which is at time = self.DT
                    x_current = solution.y[:, -1]
                    self.x = np.vstack((self.x, x_current))
                    # vertically stack u
                    self.u = np.vstack((self.u, u_current))

            return self.x, self.u

    def plot_traj(self, x, u):
        fig, ax = plt.subplots()

        circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]),
                            obstacle_list[0][2], color='r', alpha=0.2)
        ax.add_artist(circle)
        ax.plot(x[:, 0], x[:, 1])
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()


if __name__ == "__main__":
    # set up an obstacle with 0.5 m radius
    obstacle_list = [[2.9, 2.6, 0.5]]
    CBF_Planning = CBF(obstacle_list)

    with_QP = False
    without_QP = True

    # Test four examples
    # Case 1
    initial_state = np.array([1.0, 1.0, np.pi/4])
    CBF_Planning.set_initial_state(initial_state)
    u_ref = [0.0 for _ in range(50)]

    if with_QP:
        x, u = CBF_Planning.motion_planning_with_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)
    if without_QP:
        x, u = CBF_Planning.motion_planning_without_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)

    # Case 2
    initial_state = np.array([1.0, 1.0, np.pi/4])
    CBF_Planning.set_initial_state(initial_state)
    u_ref = [0.1 for _ in range(50)]

    if with_QP:
        x, u = CBF_Planning.motion_planning_with_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)
    if without_QP:
        x, u = CBF_Planning.motion_planning_without_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)

    # Case 3
    initial_state = np.array([0.0, 0.0, 0.0])
    CBF_Planning.set_initial_state(initial_state)
    u_ref = [0.3 for _ in range(50)]

    if with_QP:
        x, u = CBF_Planning.motion_planning_with_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)
    if without_QP:
        x, u = CBF_Planning.motion_planning_without_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)

    # Case 4
    initial_state = np.array([1.0, 1.0, 0.0])
    CBF_Planning.set_initial_state(initial_state)
    u_ref = [0.5 for _ in range(50)]

    if with_QP:
        x, u = CBF_Planning.motion_planning_with_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)
    if without_QP:
        x, u = CBF_Planning.motion_planning_without_QP(
            u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x, u)
