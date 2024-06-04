import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from egm.EGMsocket import *
from misc_scripts.weight_getter import *
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import quat2mat, mat2quat, axangle2quat
from transforms3d.axangles import mat2axangle, axangle2mat
from transforms3d.euler import mat2euler, euler2mat, quat2euler


class SimRobot(gym.Env):
    def __init__(self, obv_space_dim, act_space_dim, state_count, state_size):
        super(SimRobot, self).__init__()
        self.obv_space_dim = obv_space_dim
        self.act_space_dim = act_space_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obv_space_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(act_space_dim,), dtype=np.float32
        )

        # state var
        self.state_count = state_count
        self.state_size = state_size
        self.current_state = np.zeros((state_count*state_size-1,), dtype=np.float32)
        self.prev_time = time.time()
        self.pos_wobj0_to_receptacle = np.array([600,-200,70]) # input in mm
        self.quat_wobj0_to_receptacle = np.array(euler2quat(0,0,np.pi/2)) # input in quaternion
        self.T_wobj0_to_receptacle = compose(self.pos_wobj0_to_receptacle, quat2mat(self.quat_wobj0_to_receptacle),[1,1,1])
        self.T_reservoir_to_ee = compose([0,0,0], euler2mat(np.pi,0,0),[1,1,1]) # input

        # action var
        x_range = [0,120] # input in mm
        z_range = [70, 200] # input
        theta_range = [-np.pi/2, 0] # input in radians
        self.x_min = x_range[0] #+ self.pos_wobj0_to_receptacle[0]
        self.x_max = x_range[1] #+ self.pos_wobj0_to_receptacle[0]
        self.z_min = z_range[0] #+ self.pos_wobj0_to_receptacle[2]
        self.z_max = z_range[1] #+ self.pos_wobj0_to_receptacle[2]
        self.theta_min = theta_range[0]
        self.theta_max = theta_range[1]
        # print(self.x_min, self.x_max, self.z_min, self.z_max, self.theta_min, self.theta_max)
        # print(self.realize_action([0,0,0]), self.realize_action([1,1,1]))

        # reward var
        self.target_weight = 20 # grammes
        self.weight_difference_penalty = 1
        self.spillage_penalty = 5
        self.time_penalty = 1
        self.start_time = None
        termination_weight_percent = 1
        self.termination_weight_error = self.target_weight * termination_weight_percent / 100
        self.reward_floor = -5000
        self.total_spill = 0
        self.max_time = 30 # input

        # reset var
        self.start_T_wobj0_to_reservoir = compose([600,-140, 400], (euler2mat(0,0,np.pi/2)),[1,1,1]) # input

        # connection var
        computer_ip= "127.0.0.1" #"192.168.12.69"
        robot_port=6510
        self.num=0
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_socket.bind((computer_ip, robot_port))
        _, self.addr = self.robot_socket.recvfrom(1024)

        # sim_var
        self.watersim = WaterSimulator()

        self.i = 0


    def reset(self, seed=64209, options=None):
        super().reset(seed=seed, options=options)
        T_wobj_to_ee = self.start_T_wobj0_to_reservoir @ self.T_reservoir_to_ee
        pos_wobj0_to_ee, mat_wobj0_to_ee, _, _ = decompose(T_wobj_to_ee)
        x_0, z_0, theta_0 = self.pose_to_param(pos_wobj0_to_ee, mat_wobj0_to_ee, override=True)
        a,b = self.param_to_pose(x_0, z_0, theta_0, move=True)
        while True:
            x_bar, z_bar, theta_bar = self.pose_to_param()
            self.param_to_pose(x_0, z_0, theta_0, move=True)
            if np.allclose([x_0, z_0, theta_0], [x_bar, z_bar, theta_bar], atol=1e-2):
                break
        self.start_time = time.time()
        self.current_state = np.zeros((self.state_count*self.state_size-1,), dtype=np.float32)
        self.current_state[1:4] = [x_0, z_0, theta_0] 
        self.watersim = WaterSimulator()
        self.total_spill = 0
        return self.current_state, {}
    
    def pose_to_param(self, pos_wobj0_to_ee=[], mat_wobj0_to_ee=[], override=False):
        # pose should be the transform from wobj0 frame to reservoir frame
        # get pose from receptor frame to reservoir frame
        if not override: 
            pos_wobj0_to_ee, quat_wobj0_to_ee = egm_state_getter(self.robot_socket)
            mat_wobj0_to_ee = quat2mat(quat_wobj0_to_ee)
        T_wobj0_to_ee = compose(pos_wobj0_to_ee, mat_wobj0_to_ee, [1,1,1])
        T_wobj0_to_reservoir = T_wobj0_to_ee @ np.linalg.inv(self.T_reservoir_to_ee)
        T_receptacle_to_reservoir = np.linalg.inv(self.T_wobj0_to_receptacle) @ T_wobj0_to_reservoir
        pos_receptacle_to_reservoir, mat_receptacle_to_reservoir, _, _ = decompose(T_receptacle_to_reservoir)
        x,angle_receptacle_to_reservoir,z= mat2euler(mat_receptacle_to_reservoir)
        # TODO: Move the assertions to init?
        # assert(np.isclose(axis_receptacle_to_reservoir[1],1, 0.01) or np.isclose(axis_receptacle_to_reservoir[1],-1, 0.01))
        # assert(np.isclose(pos_receptacle_to_reservoir[1],0, 0.005))
        # return x, z, theta
        return pos_receptacle_to_reservoir[0], pos_receptacle_to_reservoir[2], angle_receptacle_to_reservoir
        
    def param_to_pose(self, x, z, theta, move=False):
        mat_receptacle_to_reservoir = axangle2mat([0,1,0], theta)
        pos_receptacle_to_reservoir = [x,0,z]
        T_receptacle_to_reservoir = compose(pos_receptacle_to_reservoir, mat_receptacle_to_reservoir, [1,1,1])
        T_wobj_to_reservoir = self.T_wobj0_to_receptacle @ T_receptacle_to_reservoir
        T_wobj_to_ee = T_wobj_to_reservoir @ self.T_reservoir_to_ee
        pos_wobj0_to_ee, mat_wobj0_to_ee, _, _ = decompose(T_wobj_to_ee)
        quat_wobj_to_ee = mat2quat(mat_wobj0_to_ee)
        if move:
            # print(f"Received {[x,z, theta]} and moving to ", pos_wobj0_to_ee, quat2euler(quat_wobj_to_ee))
            egm_state_sender(self.robot_socket, self.addr, pos_wobj0_to_ee, quat_wobj_to_ee, self.num)
            a,b=(egm_state_getter(self.robot_socket))
            self.num += 1
        return pos_wobj0_to_ee, quat_wobj_to_ee
    
    def process_next_state(self, w, x, z, theta, prev_state):
        t = time.time() - self.prev_time
        self.prev_time = time.time()
        obv_state = np.concatenate(([w,x,z,theta,t], prev_state))
        obv_state = obv_state[:-5] # remove last 5
        return obv_state
    
    def realize_action(self, action):
        x_real = (action[0]) * (self.x_max - self.x_min) + self.x_min
        z_real = (action[1]) * (self.z_max - self.z_min) + self.z_min
        theta_real = (action[2]) * (self.theta_max - self.theta_min) + self.theta_min
        
        return x_real, z_real, theta_real
    
    def step(self, action):
        """
        Unlike a simulated rl, the step should not process
        next_state should be taken from robot
        """
        # use action:
        self.param_to_pose(*self.realize_action(action), move=True)
        # update state
        x_dot, z_dot, theta_dot = self.pose_to_param()
        weight_dot, spill_dot = self.watersim.step(x_dot,z_dot,-theta_dot,25)
        self.total_spill += spill_dot
        current_weight = self.current_state[0] + weight_dot
        self.current_state = self.process_next_state(current_weight, x_dot, z_dot, theta_dot, self.current_state.copy())

        # define reward
        reward = (
            -self.weight_difference_penalty * np.abs(current_weight-self.target_weight) +
            -self.spillage_penalty * self.total_spill + # replace with spill value
            -self.time_penalty * (time.time() - self.start_time)
        )

        terminated = False
        truncated = False
        if (abs(theta_dot) < 0.2 and 
            current_weight >= self.target_weight - self.termination_weight_error):
            print(f"Pass, with reward {reward}, and weight {current_weight}")
            terminated = True
            truncated = False
        elif ((reward < self.reward_floor) or 
            current_weight > 2 * self.target_weight or
            (time.time() - self.start_time) > self.max_time):
            print(f"Fail with reward {reward}")
            reward = -1e4
            terminated = True
            truncated = True
        info = {}

        # self.i += 1
        # if self.i == 500:
        #     print("Weight", current_weight, self.total_spill, theta_dot,abs(theta_dot) < 0.2, current_weight >= self.target_weight - self.termination_weight_error)
        #     self.i = 0

        return (
            self.current_state.copy().astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

