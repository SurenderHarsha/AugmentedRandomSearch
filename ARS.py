# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:56:44 2019

@author: admin
"""

from NeuroEvolve import *
import numpy as np
import os


class HyperParameters():
    def __init__(self,nb_steps=1000,epsiode_length=1000,learning_rate=0.02,nb_directions=16,nb_best_directions=16,noise=0.03,seed=1):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        #self.env_name = 'BipedalWalker-v2'

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class ARSModel():
    def __init__(self,input_size,output_size,neural_net=None):
        
        if neural_net==None:
            self.nn=NeuroES(input_size,output_size,linear)
            self.nn.completed_network()
        else:
            self.nn=neural_net
        self.weight_size=self.nn.get_weight_count()
        self.theta = np.zeros(self.weight_size)
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            self.nn.set_weights(self.theta)
            return self.nn.evaluate(input)
            #return self.theta.dot(input)
        elif direction == "positive":
            
            self.nn.set_weights(self.theta + hp.noise*delta)
            return self.nn.evaluate(input)
        else:
            self.nn.set_weights(self.theta - hp.noise*delta)
            return self.nn.evaluate(input)
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
    def save_model(self,name):
        import pickle
        file = open(name+".mdl", 'wb')

        pickle.dump(self.nn, file)
        file=open(name+'.wts',"wb")
        pickle.dump(self.theta,file)
        
    def load_model(self,name):
        
        import pickle
        file = open(name+".mdl", 'rb')

        self.nn=pickle.load(file)
        file=open(name+'.wts',"rb")
        self.theta=pickle.dump(file)
def explore(env, normalizer, model, direction = None, delta = None, show=None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = model.evaluate(state, delta, direction)
        
        state, reward, done, _ = env.step(action)
        if show!=None:
            env.render()
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards



class ARSTrain():
    def __init__(self,env,normalizer,model,hp):
        global hp
        for step in range(hp.nb_steps):
        
            # Initializing the perturbations deltas and the positive/negative rewards
            deltas = model.sample_deltas()
            positive_rewards = [0] * hp.nb_directions
            negative_rewards = [0] * hp.nb_directions
            
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, normalizer, model, direction = "positive", delta = deltas[k])
            
            # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, normalizer, model, direction = "negative", delta = deltas[k])
            
            # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()
            
            # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
            
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
            
            # Updating our policy
            model.update(rollouts, sigma_r)
            
            # Printing the final reward of the policy after the update
            reward_evaluation = explore(env, normalizer, model)
            print('Step:', step, 'Reward:', reward_evaluation)
    