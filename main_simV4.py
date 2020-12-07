#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:05 2020

@author: yannis
"""

import torch
import random
from pdb import set_trace as bp
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import  get_vec_normalize
import motion_imitation
import time
import numpy as np



def testPolicy(path,scales=None,pol_scales=None):

    
    processes = 1

    

    render = True
    


    seed = 1

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_vec_envs(
        'A1GymEnv-v1',
        seed,
        processes,
        None,
        None,
        device='cpu',
        allow_early_resets=True, render=render)

    env_core = env.venv.venv.envs[0].env.env
    actor_critic,  ob_rms = torch.load(path,map_location=torch.device('cpu'))
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, processes)
    #env_core = env.venv.venv.envs[0]
    if processes==1:
        N_sim = 100
        Reward = np.zeros((N_sim,))
        input('press enter')
        n=0
        R=0
        obs=env.reset()
        while n<N_sim: 
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales)
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(obs,recurrent_hidden_states,masks, deterministic = True )   
            obs, reward, done, _ = env.step(action[0])
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales)
            #env_core.cam_track_torso_link()
            R+=reward
            #control_steps +=1
            time.sleep(5*1.0/240.0)
            if done:
                n+=1
                Reward[n]=R
                print('Reward: ',R)
                R=0
                #obs=env.reset()
                #obs[:,-4:] = torch.FloatTensor(pol_scales)
                #input('press enter')
            
            masks.fill_(0.0 if done else 1.0)
        #print('Scale: ', Scale[j,:], ', total reward:' , Reward)
        input('press enter')
    else:
        N_sim = processes
        TotalReward = np.zeros((processes,))   
        obs=env.reset()
        #bp()
        n = 0
        while n<N_sim:
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy

            TotalReward += reward.numpy().flatten()
            for D in done:
                if D:
                    #print(done)
                    n+=1
            masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        print('TotalReward: ', TotalReward, flush=True)
        AverageTotalReward = np.mean(TotalReward)
        Std = np.std(TotalReward)
        #print(TotalReward)
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std,', virtual scale: ', obs[0,-4:], flush=True)

    #bp()

        N_sim = processes
        TotalReward = np.zeros((processes,))   
        obs=env.reset()
        #bp()
        #bp()
        n = 0
        while n<N_sim:
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy
            with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=True)

            obs, reward, done, _ = env.step(action)
            if pol_scales is not None:
                obs[:,-4:] = torch.FloatTensor(pol_scales) # replace scale in the input of the policy

            TotalReward += reward.numpy().flatten()
            for D in done:
                if D:
                    #print(done)
                    n+=1
            masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        print('TotalReward: ', TotalReward, flush=True)
        AverageTotalReward = np.mean(TotalReward)
        Std = np.std(TotalReward)
        #print(TotalReward)
        print('Av. Total reward: ',AverageTotalReward, ', std: ',Std,', virtual scale: ', obs[0,-4:], flush=True)
    env.close()
#bp()


if __name__ == '__main__':
    scales = None
    pol_scales = None
    path = '/home/yannis/Repositories/motion_imitation/12_03_nominal_policy/ppo/A1GymEnv-v1.pt'
    
    testPolicy(path,scales,pol_scales)