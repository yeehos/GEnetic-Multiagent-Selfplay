# To prevent the code from grabbing too much resource
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 


import torch
import numpy as np
import time
import gym
import subprocess

from PPO import PPO
import argparse

import signal
import sys
import shutil

def exit_program(sig, frame):
    print('Exiting...')
    shutil.rmtree('temp'+args.save_dir)
    sys.exit(0)
signal.signal(signal.SIGINT, exit_program)

################################### Training ###################################
def train():

    #Training Specific Parameters
    max_ep_len = args.max_ep_len
    
    #PPO Hyperparameters
    action_std = args.action_std                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = args.action_std_decay_rate        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = args.min_action_std                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = args.action_std_decay_freq  # action_std decay frequency (in num timesteps)
    update_timestep = args.hz      # update policy every n timesteps
    K_epochs = args.ep               # update policy for K epochs in one PPO update
    eps_clip = args.eps_clip          # clip parameter for PPO
    gamma = args.gamma            # discount factor
    lr_actor = args.lr       # learning rate for actor network
    lr_critic = args.lrc       # learning rate for critic network
    
    
    #Environment
    from sumo_env import Sumo
    env = Sumo('sumo_ant',victory_reward=args.victory_reward,timeout_reward=args.timeout_reward,lose_reward=args.lose_reward,time_reward=args.time_reward,w_CEN = args.w_CEN, w_TGT = args.w_TGT, w_PSH = args.w_PSH)
    obs_dim = 136
    act_dim = 8                
    
    # Get Action
    def get_action(o, player=0,evaluate=False):                        
        return AGENTS[player].select_action(o,evaluate) 

    # Generate a random scenario
    num_policies = 0
    def get_scenario():        
        genome = []    
        
        # Environment Parameter
        genome.append(np.random.uniform(-1,1))
           
        # Opponent Codes
        # 4 : Open Loop Opponent
        # 5+ : Historical Opponents
        scenario_limit = 5+num_policies
        lower_limit = 4
        genome.append(np.random.randint(lower_limit,scenario_limit))
        
        return genome
    
    def evaluate_agent(n=100):
        # Define controller as evaluation mode
        for agent in AGENTS:
            agent.set_evaluation()

        win_list = []
        tie_list = []
        lose_list = []
        
        for i in range(n):                    
            scenario = get_scenario()
            
            o, r, d, ep_ret, ep_len = env.reset(scenario), 0, False, 0, 0                                    
                
            while not(d or (ep_len == max_ep_len)):                
                a = get_action(o,evaluate=True)                
                
                ENY_o = env.ENY_obs                                
                ENY_a = get_action(ENY_o,1,evaluate=True)
                
                o, r, d, info = env.step(a,ENY_a)                                                            
                ep_ret += r
                ep_len += 1                
                
                if d:
                    if env.victory:
                        win_list.append(1)
                        tie_list.append(0)
                        lose_list.append(0)
                    elif env.ENY_victory:
                        win_list.append(0)
                        tie_list.append(0)
                        lose_list.append(1)
                    else:
                        win_list.append(0)
                        tie_list.append(1)
                        lose_list.append(0)
                    break

        # Define agents in training mode
        for agent in AGENTS:
            agent.set_training()

        print('WIN-TIE-LOSE',round(np.mean(win_list)*100,3),'%',round(np.mean(tie_list)*100,3),'%',round(np.mean(lose_list)*100,3),'%')        
        return

    # Genetic Operations
    def mate(parent0,parent1):

        # Splitting between fixed-length heads and variable length tails
        parent0_head = parent0[:2]
        parent1_head = parent1[:2]
        parent0_tail = parent0[2:]
        parent1_tail = parent1[2:]

        offspring0 = [parent0_head[0]]
        offspring1 = [parent1_head[0]]

        if np.random.uniform()<0.5:
            offspring0.append(parent1_head[1])
            offspring1.append(parent0_head[1])
        else:
            offspring0.append(parent0_head[1])
            offspring1.append(parent1_head[1])

        parent0_tail_spilt = np.random.randint(len(parent0_tail)+1)
        parent0_tail0 = parent0_tail[:parent0_tail_spilt]
        parent0_tail1 = parent0_tail[parent0_tail_spilt:]

        parent1_tail_spilt = np.random.randint(len(parent1_tail)+1)
        parent1_tail0 = parent1_tail[:parent1_tail_spilt]
        parent1_tail1 = parent1_tail[parent1_tail_spilt:]
        
        [offspring0.append(element) for element in parent0_tail0]
        [offspring0.append(element) for element in parent1_tail1]
        
        [offspring1.append(element) for element in parent1_tail0]
        [offspring1.append(element) for element in parent0_tail1]
        return offspring0, offspring1
    
    # CrossOver Operation
    def crossover(seed_population,VLGA_pop_size,p = []):        
        genetic_population = []
        if len(seed_population)!=0:    
            while len(genetic_population)<VLGA_pop_size:
                if len(p)==0:
                    parent0_idx = np.random.choice(len(seed_population))
                    parent1_idx = np.random.choice(len(seed_population))
                else:                    
                    parent0_idx = np.random.choice(len(seed_population),p=p)                    
                    parent1_idx = np.random.choice(len(seed_population),p=p)

                parent0,parent1 = seed_population[parent0_idx],seed_population[parent1_idx]                
                offspring0, offspring1 = mate(parent0, parent1)
                #Add to Pop
                genetic_population.append(offspring0)
                genetic_population.append(offspring1)
                
        else:
            genetic_population = [get_scenario() for _ in range(VLGA_pop_size)]
        return genetic_population

    # Mutation Operation
    def mutation(genetic_population,mutation_rate=0.1):        
        for i in range(len(genetic_population)):
            if np.random.uniform()<mutation_rate:
                mutation_pair = get_scenario()
                new_genomes = mate(genetic_population[i],mutation_pair)
                new_genome = new_genomes[np.random.randint(2)]
                genetic_population[i] = new_genome
               
        return genetic_population

    # Any explorers still running?
    def active_process():
        for pid in range(args.num_proc_exp):
            if os.path.exists('temp'+args.save_dir+'/'+str(int(pid))+'running.npy'):            
                return True
        return False
    
    #Generate Curriculum
    def generate_curriculum(VIC_list,R_list,curriculum):        
        
        # Processing the outcomes
        sub_VIC_list = []
        for idx in range(len(VIC_list)):
            dice = np.random.uniform()
            if dice<=np.mean(VIC_list[idx]):                    
                sub_VIC_list.append(1)
            else:
                sub_VIC_list.append(0)
        if np.sum(sub_VIC_list) == 0:
            sub_VIC_list = [1 for _ in range(len(VIC_list))]
        
        # Inheriting the previous curriculum as seed population
        seed_population = []
        mate_probability = []
        for idx in range(len(curriculum)):
            if sub_VIC_list[idx] == 1:
                seed_population.append(curriculum[idx])                            
                mate_probability.append(np.mean(R_list[idx]))                                        
        curriculum = seed_population[:]

        # Calculating mate probability
        if np.sum(mate_probability)==0:
            mate_probability = [1 for _ in mate_probability]        
        mate_probability = np.array(mate_probability)/sum(mate_probability)
        
        # Generating the next curriculum
        while len(curriculum)<args.train_size:                        
            population = crossover(seed_population,args.train_size-len(seed_population),mate_probability)                    
            population = mutation(population,mutation_rate=args.mutation_rate)            
            curriculum.extend(population)

        curriculum = curriculum[:args.train_size]

        return curriculum
        
            
    # initialize a PPO agent
    AGENTS = []        
    AGENTS.append(PPO(obs_dim, act_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std,network_size=args.net,activation='Tanh'))

    if args.train==0:
        # Initializing opponent agent
        AGENTS.append(PPO(obs_dim, act_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std,network_size=args.net,activation='Tanh'))    
        
        print('LOAD AGENT....')
        
        AGENTS[0].load('trained_model')
        AGENTS[1].load('trained_model')
        
        print('AGENTS LOADED')
        
        evaluate_agent(n=100)
        exit()

    # Directories for saving weights and temporary files    
    os.makedirs(args.save_dir) 
    os.makedirs('temp'+args.save_dir) 
        
        
    t = 0
    policy_idx = 0    
    AGENTS[0].save(args.save_dir+'/main'+str(int(policy_idx)))
    policy_idx += 1    
    num_policies += 1

    curriculum = [get_scenario() for _ in range(args.train_size)]
    
    # Broadcast the curriculum
    for pid in range(args.num_proc_exp):
        next_scenario = curriculum.pop(0)        
        np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario.npy',np.array(next_scenario[:2],dtype=object))
        np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario_tail.npy',np.array(next_scenario[2:],dtype=object))
    
    # Performance buffer
    SCENARIO_list = []
    VIC_list = []
    REGRET_list = []
    
    verbose = args.verbose
    
    explorer_name = 'explorer.py'
    
    #Save the model weights here!
    AGENTS[0].save(args.save_dir+'/main')            
    
    # Let agents know we ar starting
    np.save(args.save_dir+'/weight_ready.npy',np.array([0]))        
    
    if verbose: print('MAIN| LAUNCHING EXPLORERS')
    for i in range(args.num_proc_exp):        
        subprocess.Popen(['python',explorer_name,'--fall-penalty',str(args.fall_penalty),'--net',str(args.net),'--w-TGT',str(args.w_TGT),'--w-PSH',str(args.w_PSH),'--w-CEN',str(args.w_CEN),'--victory-reward',str(args.victory_reward),'--timeout-reward',str(args.timeout_reward),'--lose-reward',str(args.lose_reward),'--time-reward',str(args.time_reward),'--pid',str(i),'--num-proc-exp',str(args.num_proc_exp),'--save-dir',str(args.save_dir)],shell=False,stdin=None,stdout=None,stderr=None,close_fds=True)    
    
    if verbose: print('MAIN| ALL EXPLORERS LAUNCHED')

    policy_generation = 0
    process_scanning = 0
    decay_count = 0
    prev_epoch = 0 
    process_ready = False
    
    while True:        
        process_scanning = (process_scanning+1)%args.num_proc_exp        
        datapacket_size = 0
        try:                
            process_output = np.load('temp'+args.save_dir+'/'+str(int(process_scanning))+'_output.npz',allow_pickle=True)#
            datapacket_size +=1            
            pid_rewards = process_output['rewards']            
            pid_is_terminals = process_output['is_terminals']
            pid_states = process_output['states']                        
            pid_actions = process_output['actions']
            pid_logprobs = process_output['logprobs']            
            pid_t = process_output['t']            
            VIC = process_output['VIC']
            regret = process_output['regret']
            SCENARIO = process_output['SCENARIO'].tolist()
            SCENARIO_TAIL = process_output['SCENARIO_TAIL'].tolist()
            SCENARIO.extend(SCENARIO_TAIL)
            
            process_ready = True            
        
        except:
            process_ready = False
            pass
        
        
        #RESPOND to the output
        if process_ready:            
            
            # Erase the read file
            file_path = os.path.join('temp', args.save_dir, f"{int(process_scanning)}_output.npz")
            while os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    time.sleep(0.01)
                except:
                    pass
            
            # Outcome buffer
            SCENARIO_list.append(SCENARIO)
            if VIC==1:
                VIC_list.append(0)
            elif VIC==0:
                VIC_list.append(0.5)
            else:
                VIC_list.append(1)            
            REGRET_list.append(regret)

            # Generate a curriculum when you are out of curriculum to explore
            if len(curriculum)==0:
                if verbose: print('MAIN| GENERATING CURRICULUM')  
                curriculum = generate_curriculum(VIC_list,REGRET_list,SCENARIO_list)                
                SCENARIO_list = []
                VIC_list = []
                REGRET_list = []                
                if verbose: print('MAIN| CURRICULUM GENERATED')


            if (t+pid_t)<(policy_generation+1)*update_timestep:
                next_scenario = curriculum.pop(0)                                
                np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario.npy',next_scenario[:2])
                np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario_tail.npy',next_scenario[2:])
                
            
            # Update to PPO Buffer            
            t += pid_t
            for reward,is_terminal,state,action,logprob in zip(pid_rewards,pid_is_terminals,pid_states,pid_actions,pid_logprobs):
                AGENTS[0].buffer.rewards.append(reward)
                AGENTS[0].buffer.is_terminals.append(is_terminal)
                with torch.no_grad():
                    AGENTS[0].buffer.states.append(torch.FloatTensor(state).to())
                    AGENTS[0].buffer.actions.append(torch.FloatTensor(action).to())
                    AGENTS[0].buffer.logprobs.append(torch.FloatTensor(logprob).to())
                
        # Update PPO Network
        if not active_process() and t>(policy_generation+1)*update_timestep:               
            
            if verbose: print('MAIN| UPDATING MODEL |T',t)  
            
            # Update the PPO Network and save it as the latest model
            AGENTS[0].update()           
            AGENTS[0].save(args.save_dir+'/main')        
            policy_generation += 1
            np.save(args.save_dir+'/weight_ready.npy',np.array([policy_generation]))

            #Launch the explorations
            while len(curriculum)<args.num_proc_exp:
                curriculum.append(get_scenario()) #Buffering.....

            for pid in range(args.num_proc_exp):
                next_scenario = curriculum.pop(0)
                np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario.npy',next_scenario[:2])
                np.save('temp'+args.save_dir+'/'+str(int(pid))+'_scenario_tail.npy',next_scenario[2:])

            process_scanning = 0                        
            while t>(action_std_decay_freq*(decay_count+1)):
                decay_count+=1
                AGENTS[0].decay_action_std(action_std_decay_rate, min_action_std)
            
        
        # Save policy weights to the library
        epoch = int(t/args.training_steps)        
        
        if epoch>prev_epoch:            
            prev_epoch=epoch            
        
            AGENTS[0].save(args.save_dir+'/main')    
            AGENTS[0].save(args.save_dir+'/main'+str(int(policy_idx)))
            policy_idx += 1    
            num_policies += 1
            
            print('=======================================')
            print('Epoch :',epoch)                                             
            print('Steps :',"{:e}".format(t+1))   
            print('=======================================')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    # Training Parameters
    parser.add_argument('--training-steps',  type=int, default=int(1e5))   
    parser.add_argument('--save-dir', type=str, default='saved_models')    
    parser.add_argument('--load-dir',type=str,default='GEMS_trained')
    parser.add_argument('--train',type=int,default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)    
    parser.add_argument('--verbose',type=int,default=0)
    parser.add_argument('--hz', type=int, default=36864)
    parser.add_argument('--action-std', type=float, default=0.6)
    parser.add_argument('--action-std-decay-rate', type=float, default=0.05)
    parser.add_argument('--min-action-std', type=float, default=0.1)
    parser.add_argument('--action-std-decay-freq', type=int, default=int(2.5e5))
    parser.add_argument('--eps-clip', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lrc', type=float, default=0.001)
    parser.add_argument('--update-timestep', type=int, default=2048)
    parser.add_argument('--net', type=int, default=64)
    parser.add_argument('--ep', type=int, default=20)
    

    # Environment Setup    
    parser.add_argument('--victory-reward',type=float,default=300)    
    parser.add_argument('--timeout-reward',type=float,default=0)
    parser.add_argument('--lose-reward',type=float,default=-100)
    parser.add_argument('--time-reward',type=float,default=0)
    parser.add_argument('--fall-penalty',type=float,default=0)
    parser.add_argument('--w-TGT',type=float,default=3)
    parser.add_argument('--w-PSH',type=float,default=10)
    parser.add_argument('--w-CEN',type=float,default=1)

    
    # Curriculum Generation        
    parser.add_argument('--mutation-rate',type=float,default=0.1)
    parser.add_argument('--train-size',type=int,default=200)
    
    # Parallelization
    parser.add_argument('--num-proc-exp',type=int,default=2)
    
    args = parser.parse_args()
    
    

    train()
    
    
    
    
    
    
    
