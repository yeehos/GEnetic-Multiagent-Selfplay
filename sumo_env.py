import gym
import gym_compete
import numpy as np
class Sumo():
    def __init__(self,env_type='sumo_ant',victory_reward=1000,timeout_reward=1000,w_CEN=0,w_TGT=0,w_PSH=0,lose_reward=-1000,fall_penalty=0,time_reward=0):
        self.victory_bonus = victory_reward
        self.timeout_penalty = timeout_reward
        self.w_CEN = w_CEN
        self.w_TGT = w_TGT
        self.w_PSH = w_PSH
        self.lose_reward = lose_reward
        self.time_reward = time_reward
        self.env_type = env_type
        self.fall_penalty = fall_penalty
        
        self.env = gym.make("sumo-ants-v0")                    
        self.altitude = 1.1            
        self.scenario = [0]
        
    def step(self,a,ENY_a):        
        self.episode_step+=1
        
        # Simulate a step        
        
        # Who's playing red/blue?
        if self.scenario[0]>=0:
            obs,_,done,info =self.env.step([ENY_a,a])
        else:
            obs,_,done,info =self.env.step([a,ENY_a])

        obs1,obs0 = obs

        #Calculate Dense Reward
        new_pot = np.sqrt(np.sum((obs0[:2])**2))*self.w_CEN-np.sqrt(np.sum((obs1[:2])**2))*self.w_PSH+np.sqrt(np.sum([(a-b)**2 for a,b in zip(obs0[:2],obs1[:2])]))*self.w_TGT
        ENY_new_pot = np.sqrt(np.sum((obs1[:2])**2))*self.w_CEN-np.sqrt(np.sum((obs0[:2])**2))*self.w_PSH+np.sqrt(np.sum([(a-b)**2 for a,b in zip(obs1[:2],obs0[:2])]))*self.w_TGT        
        
        reward = self.pot - new_pot
        ENY_reward = self.ENY_pot - ENY_new_pot        
        
        self.pot = new_pot
        self.ENY_pot = ENY_new_pot

        victory,ENY_victory = False,False

        #Based on the outcome        
        if done[0]:            
            if ('winner' in info[0]) and not ('winner' in info[1]):                                
                ENY_victory = True
                reward += self.lose_reward
                ENY_reward += self.victory_bonus

            if ('winner' in info[1]) and not ('winner' in info[0]):                
                victory = True
                reward += self.victory_bonus
                ENY_reward += self.lose_reward

        
        reward += self.time_reward
        ENY_reward += self.time_reward

        if self.scenario[0]>=0: # Who's playing red/blue?
            self.obs = obs0[:-1]        
            self.ENY_obs = obs1[:-1]
            self.victory = victory
            self.ENY_victory = ENY_victory
            self.reward = reward
            self.ENY_reward = ENY_reward
        else:
            self.obs = obs1[:-1]        
            self.ENY_obs = obs0[:-1]
            self.victory = ENY_victory
            self.ENY_victory = victory
            self.reward = ENY_reward
            self.ENY_reward = reward

        return self.obs, self.reward, done[0], [[],self.victory,_,_,[],[]]

    def reset(self,scenario=[0,0]):
        
        self.scenario = scenario[:2]
        self.env.radius = 1
        pos = self.env.reset(scenario=scenario)#We don't send reward data to the sumo since that part is done by CSP_env*.py
        try:
            pos = self.env.reset(scenario=scenario)#We don't send reward data to the sumo since that part is done by CSP_env*.py
        except:
            print('FAILED TO RESET WITH',scenario)
            exit()
        
        if self.scenario[0]>=0:
            self.ENY_obs,self.obs = pos[0][:-1],pos[1][:-1]            
        else:
            self.ENY_obs,self.obs = pos[1][:-1],pos[0][:-1]
            
        obs0 = pos[1][:-1]
        obs1 = pos[0][:-1]
        
        
        self.episode_step = 0
        
        
        self.pot = np.sqrt(np.sum((obs0[:2])**2))*self.w_CEN-np.sqrt(np.sum((obs1[:2])**2))*self.w_PSH+np.sqrt(np.sum([(a-b)**2 for a,b in zip(obs0[:2],obs1[:2])]))*self.w_TGT
        self.ENY_pot = np.sqrt(np.sum((obs1[:2])**2))*self.w_CEN-np.sqrt(np.sum((obs0[:2])**2))*self.w_PSH+np.sqrt(np.sum([(a-b)**2 for a,b in zip(obs1[:2],obs0[:2])]))*self.w_TGT
        return self.obs
    
    def render(self,mode='rgb_array'):
        if mode=='rgb_array':            
            return self.env.render(mode='rgb_array')
        else:
            print('WE CURRENTLY ONLY SUPPORT mode=rgb_array')
            exit()




    