from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pdb import set_trace as bp
#DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
#(0, b'imu_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'imu_link', (0.0, 0.0, 0.0), (-0.012731, -0.002186, -0.000515), (0.0, 0.0, 0.0, 1.0), -1)
#(1, b'FR_hip_joint', 0, 7, 6, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'FR_hip', (1.0, 0.0, 0.0), (0.170269, -0.049186, -0.000515), (0.0, 0.0, 0.0, 1.0), -1)
#(2, b'FR_hip_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FR_upper_shoulder', (0.0, 0.0, 0.0), (0.003311, -0.080365, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 1)
#(3, b'FR_upper_joint', 0, 8, 7, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'FR_upper', (0.0, 1.0, 0.0), (0.003311, -0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 1)
#(4, b'FR_lower_joint', 0, 9, 8, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'FR_lower', (0.0, 1.0, 0.0), (0.003237, -0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 3)
#(5, b'FR_toe_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FR_toe', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 4)
#(6, b'FL_hip_joint', 0, 10, 9, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'FL_hip', (1.0, 0.0, 0.0), (0.170269, 0.044814, -0.000515), (0.0, 0.0, 0.0, 1.0), -1)
#(7, b'FL_hip_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FL_upper_shoulder', (0.0, 0.0, 0.0), (0.003311, 0.080365, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 6)
#(8, b'FL_upper_joint', 0, 11, 10, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'FL_upper', (0.0, 1.0, 0.0), (0.003311, 0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 6)
#(9, b'FL_lower_joint', 0, 12, 11, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'FL_lower', (0.0, 1.0, 0.0), (0.003237, 0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 8)
#(10, b'FL_toe_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FL_toe', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 9)
#(11, b'RR_hip_joint', 0, 13, 12, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'RR_hip', (1.0, 0.0, 0.0), (-0.195731, -0.049186, -0.000515), (0.0, 0.0, 0.0, 1.0), -1)
#(12, b'RR_hip_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RR_upper_shoulder', (0.0, 0.0, 0.0), (-0.003311, -0.080365, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 11)
#(13, b'RR_upper_joint', 0, 14, 13, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'RR_upper', (0.0, 1.0, 0.0), (-0.003311, -0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 11)
#(14, b'RR_lower_joint', 0, 15, 14, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'RR_lower', (0.0, 1.0, 0.0), (0.003237, -0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 13)
#(15, b'RR_toe_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RR_toe', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 14)
#(16, b'RL_hip_joint', 0, 16, 15, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'RL_hip', (1.0, 0.0, 0.0), (-0.195731, 0.044814, -0.000515), (0.0, 0.0, 0.0, 1.0), -1)
#(17, b'RL_hip_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RL_upper_shoulder', (0.0, 0.0, 0.0), (-0.003311, 0.080365, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 16)
#(18, b'RL_upper_joint', 0, 17, 16, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'RL_upper', (0.0, 1.0, 0.0), (-0.003311, 0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 16)
#(19, b'RL_lower_joint', 0, 18, 17, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'RL_lower', (0.0, 1.0, 0.0), (0.003237, 0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 18)
#(20, b'RL_toe_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RL_toe', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 19)

class PPOLocomotionTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.alive_bonus = 4.5
    self.max_tar_vel = 2.5
    self.vel_r_weight = 4.0
    self.ctrl_dofs = np.array([ 1, 3, 4, 6, 8,  9, 11, 13, 14, 16, 18, 19 ])
    self.jl_weight = 0.5
    self.dq_pen_weight = 0.001
    
    #self.ul = np.array([0.,-0.802851455917, 0. , -1.0471975512, -2.69653369433, 0. , -0.802851455917, 0., -1.0471975512, -2.69653369433, 0., -0.802851455917, 0., -1.0471975512, -2.69653369433, 0., -0.802851455917, 0., -1.0471975512, -2.69653369433, 0.0])
    #self.ll = np.array([-1., 0.802851455917, -1.0, 4.18879020479, -0.916297857297, -1.0, 0.802851455917, -1.0, 4.18879020479, -0.916297857297, -1.0, 0.802851455917, -1.0, 4.18879020479, -0.916297857297, -1.0, 0.802851455917, -1.0, 4.18879020479, -0.916297857297, -1.0])

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos
    



  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    return rot_mat[-1] < 0.85

  def get_q_dq(self, env,dofs):
    joints_state = env._pybullet_client.getJointStates(env.robot.quadruped, dofs)
    joints_q = np.array(joints_state)[:, [0]]
    joints_q = np.hstack(joints_q.flatten())
    joints_dq = np.array(joints_state)[:, [1]]
    joints_dq = np.hstack(joints_dq.flatten())
    return joints_q, joints_dq


  def reward(self, env):
    """Get the reward without side effects."""
    dt = env._env_time_step 
    timer = env.robot._state_action_counter
    quadruped = env.robot.quadruped 
    ll = np.array([env._pybullet_client.getJointInfo(quadruped, i)[8] for i in self.ctrl_dofs])
    ul = np.array([env._pybullet_client.getJointInfo(quadruped, i)[9] for i in self.ctrl_dofs])



    reward = self.alive_bonus

    
    velx = (self.current_base_pos[0] - self.last_base_pos[0])/dt
    tar = np.minimum(0.1*timer / 500, self.max_tar_vel)
    reward += np.minimum(velx, tar) * self.vel_r_weight
    # print("v", self.velx, "tar", tar)
    #reward += -self.energy_weight * np.square(a).sum() #TODO missing
    # print("act norm", -self.energy_weight * np.square(a).sum())

    q, dq = self.get_q_dq(env,self.ctrl_dofs)
    pos_mid = 0.5 * (ll + ul)
    q_scaled = 2 * (q - pos_mid) / (ul - ll)
    joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
    reward += -self.jl_weight * joints_at_limit

    # print("jl", -self.jl_weight * joints_at_limit)

    reward += -np.minimum(np.sum(np.square(dq)) * self.dq_pen_weight, 5.0)
    #weight = np.array([2.0, 1.0, 1.0] * 4) #TODO missing
    #reward += -np.minimum(np.sum(np.square(q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)
    del env
    return reward