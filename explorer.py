import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy as np
import time
import gym

from PPO import PPO
import argparse

################################### Training ###################################
def train():

    #Training Specific Parameters
    max_ep_len = args.max_ep_len
    
    #PPO Hyperparameters
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)    
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)        
    K_epochs = args.ep               # update policy for K epochs in one PPO update
    eps_clip = 0.1          # clip parameter for PPO
    gamma = 0.98            # discount factor
    lr_actor = args.lr       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    

    # Environment Setup
    from sumo_env import Sumo
    env = Sumo('sumo_ant',victory_reward=args.victory_reward,timeout_reward=args.timeout_reward,lose_reward=args.lose_reward,time_reward=args.time_reward,w_CEN = args.w_CEN, w_TGT = args.w_TGT, w_PSH = args.w_PSH)
    obs_dim = 136
    act_dim = 8        
    

    def get_action(o, player=0,evaluate=False):                        
        try:
            return AGENTS[player].select_action(o,evaluate) #Evaluate is the option to use during training if doing self-self
        except:
            return np.random.uniform(-1,1,act_dim)

    def get_V(o):
        return AGENTS[0].evaluate(o)
    
    def get_regret(TD_list):
        T = len(TD_list)
        TD_sum = 0
        for k in range(T):
            TD_sum += max([0,gamma**(k-T)*TD_list[k]])
        return TD_sum/T


    
    # initialize PPO agents
    AGENTS = []        
    AGENTS.append(PPO(obs_dim, act_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std,network_size=args.net,activation='Tanh'))
    AGENTS.append(PPO(obs_dim, act_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std,network_size=args.net,activation='Tanh'))
    

    #Initialize Environment
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    loaded_opponent = -1
    MASTER_ARM = True
    GENERATION = - 1
    decay_count = 0 
    total_t = 0
    
    INITIALIZED = False
    while MASTER_ARM:
        
        # The explorers will shut down if the temporary directory to dump results is no longer existant.
        if os.path.exists('temp'+args.save_dir):
            MASTER_ARM = True
        else:
            MASTER_ARM = False
        
        if not INITIALIZED:
            # Load data from the main
            while True:
                try:
                    current_generation = np.load(args.save_dir+'/weight_ready.npy',allow_pickle=True)[0]                    
                    scenario = np.load('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy',allow_pickle=True).tolist()                                        
                    scenario_tail = np.load('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy',allow_pickle=True).tolist()          
                    scenario.extend(scenario_tail)
                    break
                except:
                    pass 
            
            # Erase files that has been read
            while os.path.exists('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy'):
                try:
                    os.remove('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy')  
                    time.sleep(0.01)
                except:                    
                    pass
            while os.path.exists('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy'):
                try:
                    os.remove('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy')  
                    time.sleep(0.01)
                except:                    
                    pass
                
            np.save('temp'+args.save_dir+'/'+str(int(args.pid))+'running.npy',int(1)) 
            
            o, r, d, ep_ret, ep_len = env.reset(scenario), 0, False, 0, 0
            R_list = []
            Q_list = []

            # Opponent Codes
            # 4 : Open Loop Opponent
            # 5+ : Historical Opponents

            if scenario[1]>4:
                if loaded_opponent!=(scenario[1]-5):                        
                    AGENTS[1].load(args.save_dir+'/main'+str(int(scenario[1]-5)))
                    loaded_opponent = scenario[1]-5
                RL_opponent = True              
            else:
                RL_opponent = False                

            if not RL_opponent:
                augmentation = [0 for _ in range(act_dim)]
                earliest_IDX = 10000
                for action_record in scenario[2:]:                    
                    if action_record[0] < earliest_IDX:
                        earliest_IDX = action_record[0]
                        augmentation = action_record[1:]                                    

            if current_generation>GENERATION:
                AGENTS[0].load(args.save_dir+'/main')            
                GENERATION = current_generation
                INITIALIZED = True                
            INITIALIZED = True

        if INITIALIZED: 
            
            # Train on the given curriculum
            while True:                                
                Q_list.append(get_V(o))    
                a = get_action(o)
                
                if RL_opponent:
                    ENY_a = get_action(o,1)
                else:
                    for action_record in scenario[2:]:                    
                        if action_record[0] == ep_len:
                            augmentation = action_record[1:]                    
                    ENY_a = augmentation        
                    
                o2, r, d, info = env.step(a,ENY_a)                                        
                
                R_list.append(r)
                AGENTS[0].update_buffer(r,d)            
                
                total_t += 1
                ep_ret += r
                ep_len += 1
                            
                o = o2           

                while total_t*args.num_proc_exp>(action_std_decay_freq*(decay_count+1)):
                    decay_count+=1
                    AGENTS[0].decay_action_std(action_std_decay_rate, min_action_std)                    

                if d or ep_len>max_ep_len:
                    # Outcome
                    if info[1]: 
                        VIC = 1
                    elif env.ENY_victory:
                        VIC = -1
                    else:
                        VIC = 0
                    
                    # Regret
                    R_list.reverse()                    
                    for i in range(len(R_list)):
                        if i>0:
                            R_list[i] += R_list[i-1]*gamma
                    R_list.reverse()
                    TD_list = np.array(R_list)-np.array(Q_list)
                    regret = get_regret(TD_list)        

                    # Reporting the data back to main                    
                    np.savez('temp'+args.save_dir+'/'+str(int(args.pid))+'_output.npz',VIC=VIC,regret=regret,t=ep_len,SCENARIO=scenario[:2],SCENARIO_TAIL=scenario[2:],rewards=AGENTS[0].buffer.rewards,is_terminals=AGENTS[0].buffer.is_terminals,states=AGENTS[0].buffer.states,actions=AGENTS[0].buffer.actions,logprobs=AGENTS[0].buffer.logprobs)                                            
                    
                    AGENTS[0].buffer.clear()
                    
                    while os.path.exists('temp'+args.save_dir+'/'+str(int(args.pid))+'running.npy'):
                        try:
                            os.remove('temp'+args.save_dir+'/'+str(int(args.pid))+'running.npy') #This file tells that we are Running!                            
                            time.sleep(0.01)
                        except:                            
                            pass

                    
                    if not os.path.exists(args.save_dir):                        
                        MASTER_ARM = False
                        break
                                
                    # Load the Next Scenario
                    while True:
                        try:                            
                            scenario = np.load('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy',allow_pickle=True).tolist()
                            scenario_tail = np.load('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy',allow_pickle=True).tolist()
                            scenario.extend(scenario_tail)
                            current_generation = np.load(args.save_dir+'/weight_ready.npy',allow_pickle=True)[0]
                            break
                        except:
                            pass         
                    
                    if current_generation>GENERATION:
                        while True:
                            try:                        
                                AGENTS[0].load(args.save_dir+'/main')            
                                GENERATION = current_generation                
                                break
                            except:
                                time.sleep(0.5)
                                pass
                    
                    while os.path.exists('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy'):
                        try:
                            os.remove('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario.npy')  
                            time.sleep(0.01)
                        except:                            
                            pass
                    while os.path.exists('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy'):
                        try:
                            os.remove('temp'+args.save_dir+'/'+str(int(args.pid))+'_scenario_tail.npy')  
                            time.sleep(0.01)
                        except:                            
                            pass
                    
                        
                    np.save('temp'+args.save_dir+'/'+str(int(args.pid))+'running.npy',int(1)) #This file tells that we are Running!              
                    
                    # Reset the Environment
                    o, r, d, ep_ret, ep_len = env.reset(scenario), 0, False, 0, 0
                    
                    R_list = []
                    Q_list = []
                    
                    # Opponent Codes
                    # 4 : Open Loop Opponent
                    # 5+ : Historical Opponents
                    
                    if scenario[1]>4:
                        if loaded_opponent!=(scenario[1]-5):                        
                            AGENTS[1].load(args.save_dir+'/main'+str(int(scenario[1]-5)))
                            loaded_opponent = scenario[1]-5
                        RL_opponent = True                      
                    else:
                        RL_opponent = False

                    if not RL_opponent:
                        earliest_IDX = 10000     
                        augmentation = [0 for _ in range(act_dim)]                   
                        for action_record in scenario[2:]:                    
                            if action_record[0] < earliest_IDX:
                                earliest_IDX = action_record[0]
                                augmentation = action_record[1:]
                        
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    # Training Parameters    
    parser.add_argument('--save-dir', type=str, default='saved_models')    
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)        
    parser.add_argument('--hz', type=int, default=2048)
    parser.add_argument('--net', type=int, default=64) 
    parser.add_argument('--ep', type=int, default=20)   
    
    # Environment Setup    
    parser.add_argument('--victory-reward',type=float,default=100)    
    parser.add_argument('--timeout-reward',type=float,default=0)
    parser.add_argument('--lose-reward',type=float,default=-20)
    parser.add_argument('--time-reward',type=float,default=-0.04)
    parser.add_argument('--fall-penalty',type=float,default=0)
    parser.add_argument('--w-TGT',type=float,default=3)
    parser.add_argument('--w-PSH',type=float,default=10)
    parser.add_argument('--w-CEN',type=float,default=1)
    
    
    # Explorer process manager
    parser.add_argument('--pid',type=int, default=0)   
    parser.add_argument('--num-proc-exp',type=int, default=2)   
    
    args = parser.parse_args()
    
    

    train()
    
    
    
    
    
    
    
