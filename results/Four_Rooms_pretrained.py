import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
import time
import numpy as np
import random
import copy
import numpy as np
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN

from agents.DQN_agents.DDQN import DDQN
from environments.Four_Rooms_Environment import Four_Rooms_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.hierarchical_agents.h_DQN import h_DQN
from scipy.special import expit
from scipy.special import logsumexp

config = Config()
config.seed = 1

height = 13
width = 13
random_goal_place = False
num_possible_states = (height * width) ** (1 + 1*random_goal_place)
embedding_dimensions = [[num_possible_states, 20]]
print("Num possible states ", num_possible_states)

config.environment = Four_Rooms_Environment(height, width, stochastic_actions_probability=0.0, random_start_user_place=True, random_goal_place=random_goal_place)

config.num_episodes_to_run = 1000
# config.file_to_save_data_results = "Data_and_Graphs/Four_Rooms.pkl"
# config.file_to_save_results_graph = "Data_and_Graphs/Four_Rooms.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = False
config.save_model = False
config.render = False


##later added info
config.discount = 0.99
config.lr_term = 0.8
config.lr_intra = 0.8
config.lr_critic = 0.8
config.nruns = 1
config.nsteps = 1000
config.noptions = 4
config.option_temperature = 1e-1
config.action_temperature = 1e-2
config.eta = 0.3
# config.new_randomness = 0.45
config.multi_option = False

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class SoftmaxPolicy:
    def __init__(self, nfeatures, nactions, temp=1.):
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        exp_v = np.exp(v - logsumexp(v))
        normalized_pmf = exp_v / np.sum(exp_v)
        return normalized_pmf
    
    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.weights)):
            v = self.value([phi])/self.temp
            all_pmfs.append(np.exp(v - logsumexp(v)))
        return np.array(all_pmfs)

    def sample(self, phi):
        prob = self.pmf(phi)
        # return int(np.random.choice(self.weights.shape[1], p=self.pmf(phi)))

        return int(np.random.choice(np.arange(self.weights.shape[1]), p = prob[0]))

class SigmoidTermination:
    def __init__(self, nfeatures):
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(np.random.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class SigmoidInterestFunction:
    def __init__(self,):
        self.room0 = list(range(5)) + list(range(10, 15)) + list(range(20, 26)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.room1 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room2 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.room3 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 89)) + list(range(94, 99))
        self.rooms = [self.room0, self.room1, self.room2, self.room3]

    def get(self, phi, option):
        interest = float(phi in self.rooms[option])
        if interest == 0.:
            interest=0.1
        return interest

    def getall(self, phi):
        interest= np.ones((4)) * 0.1
        for o in range(4):
            if phi in self.rooms[o]:
                interest[o] = 1.
        return interest

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights, meta_policy, noptions, initiation_policy=None):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.meta_policy = meta_policy
        self.noptions=noptions
        self.initiation_policy=initiation_policy

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi)
        

    def value(self, phi, option=None):
        if option is None:
            value = np.sum(self.weights[phi, :], axis=0)
        else:
            value = np.sum(self.weights[phi, option], axis=0)
        return value[0]

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - self.meta_policy.pmf(phi).dot(values)
        if option is None:
            return advantages
        return advantages[option]

    def uponarrival(self, next_phi, option):
        qvalues = self.value(next_phi,option)
        all_values = self.value(next_phi)
        if self.initiation_policy is  None:
            values = self.meta_policy.pmf(next_phi).dot(all_values)
            beta = self.terminations[option].pmf(next_phi)
        else:
            values = np.array(self.initiation_policy.pmf(next_phi)).dot(all_values)
            beta = self.terminations.pmf(next_phi,option)
        u_so = qvalues * (1-beta) + values * beta
        return u_so


    def update(self, next_phi, next_option, reward, done, multiplier):

        # One-step update target
        update_target = reward
        current_values = self.value(next_phi)
        if not done:
            update_target += self.discount * current_values[next_option]

        
        # Dense gradient update step
        tderror = update_target - self.last_value[self.last_option]


        for o in range (self.noptions):
            self.weights[self.last_phi, o] += self.lr*multiplier[o] * \
            (reward + self.discount * float(not done) * self.uponarrival(next_phi,o) - self.last_value[o] )
            

        self.last_value = self.value(next_phi)
        self.last_option = next_option
        self.last_phi = next_phi

class TerminationGradient:
    def __init__(self, terminations, critic, lr,noptions):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr
        self.noptions=noptions

    def update(self, phi, option, multiplier):
        for o in range(self.noptions):
            magnitude, direction = self.terminations[o].grad(phi)
            self.terminations[o].weights[direction] -= \
                    self.lr*multiplier[o]*magnitude*(self.critic.advantage(phi, o))             

class IntraOptionGradient:
    def __init__(self, option_policies, lr, discount, critic,noptions):
        self.lr = lr
        self.option_policies = option_policies
        self.discount = discount
        self.critic= critic
        self.noptions=noptions

    def update(self, phi, option, action, reward, done, next_phi, next_option, critic, multiplier, prob_cur_option=1.):

        for o in range(self.noptions):

            adv =(reward + self.discount * float(not done) * self.critic.uponarrival(next_phi,o) - self.critic.value(phi,o))
            actions_pmf = self.option_policies[o].pmf(phi)
            mult_o = multiplier[o]
            self.option_policies[o].weights[phi, :] -= self.lr* mult_o *prob_cur_option[o]*adv*actions_pmf
            self.option_policies[o].weights[phi, action] += self.lr* mult_o*prob_cur_option[o]*adv

class FixedTermination:
    def __init__(self,eps=0.4):
        self.room0 =  list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) 
        self.room1 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room2 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) 
        self.room3 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99))
        self.rooms = [self.room0, self.room0, self.room1, self.room1, self.room2,  self.room2, self.room3,self.room3]
        self.eps =eps

    def sample(self, phi, option):
        if option>=len(self.rooms): # Primitive actions
            termination=True
        else:                       # Options
            if phi in self.rooms[option]:
                termination = np.random.uniform() < self.eps
            else:
                termination = True
        return termination

    def pmf(self,phi, option):
        if option>=len(self.rooms): # Primitive actions
            termination_prob=1.0
        else:                       # Options
            if phi in self.rooms[option]:
                termination_prob = self.eps
            else:
                termination_prob = 1.0
        return termination_prob

class FixedInitiationSet:
    def __init__(self,):
        self.option0 = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.option1 = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) + [25]
        self.option2 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57)) + [62]
        self.option3 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57)) + [25]
        self.option4 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.option5 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [88]
        self.option6 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99)) + [51]
        self.option7 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99)) + [88]
        self.options = [self.option0, self.option1, self.option2, self.option3, self.option4,  self.option5, self.option6,self.option7]

    def get(self, phi, option):
        if option > 7:
            interest= 1.
        else:
            interest = float(phi in self.options[option])
        return interest

class InitiationSetSoftmaxPolicy:
    def __init__(self, noptions, initiationset, poveroptions):
        self.noptions = noptions
        self.initiationset = initiationset
        self.poveroptions = poveroptions

    def pmf(self, phi, option=None):
        list1 = [self.initiationset.get(phi,opt) for opt in range(self.noptions)]   
        list2 = self.poveroptions.pmf(phi)
        normalizer = sum([x * y for x, y in zip(list1, list2)])
        pmf = [float(list1[i] * list2[i])/normalizer for i in range(self.noptions)]
        return pmf

    def sample(self, phi):
        # import pdb;pdb.set_trace()
        return int(np.random.choice(self.noptions, p=self.pmf(phi)))

    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.poveroptions.weights)):
            pmf = self.pmf(np.array([phi]))
            all_pmfs.append(pmf)
        return np.array(all_pmfs)



class h_DQN(Base_Agent):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat
    Note also that this algorithm only works when we have discrete states and discrete actions currently because otherwise
    it is not clear what it means to achieve a subgoal state designated by the meta-controller"""
    agent_name = "h-DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.controller_config = copy.deepcopy(config)
        self.controller_config.output_dim_ = self.action_size
        self.controller = DDQN(self.controller_config)
        self.controller.q_network_local = self.create_NN(input_dim=self.state_size*3, output_dim=self.action_size,
                                                         key_to_use="CONTROLLER")
        self.meta_controller_config = copy.deepcopy(config)
        self.meta_controller_config.hyperparameters = self.meta_controller_config.hyperparameters["META_CONTROLLER"]
        self.meta_controller_config.magic_number = 2
        self.meta_controller_config.output_dim_ = config.environment.observation_space.n
        self.meta_controller = DDQN(self.meta_controller_config)
        self.meta_controller.q_network_local = self.create_NN(input_dim=self.state_size*2, output_dim=config.environment.observation_space.n,
                                                              key_to_use="META_CONTROLLER")
        self.rolling_intrinsic_rewards = []
        self.goals_seen = []
        self.controller_learnt_enough = False
        self.controller_actions = []

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.cumulative_meta_controller_reward = 0
        self.episode_over = False
        self.subgoal_achieved = False
        self.total_episode_score_so_far = 0
        self.meta_controller_steps = 0
        self.update_learning_rate(self.controller_config.hyperparameters["learning_rate"], self.controller.q_network_optimizer)
        self.update_learning_rate(self.meta_controller_config.hyperparameters["learning_rate"], self.meta_controller.q_network_optimizer)

    def step(self):

        self.episode_steps = 0
        copied_grid = copy.deepcopy(self.environment.grid)
        for row in range(self.environment.grid_height):
            for col in range(self.environment.grid_width):
                if copied_grid[row][col] == self.environment.wall_space_name:
                    copied_grid[row][col] = 1
                elif copied_grid[row][col] == self.environment.blank_space_name:
                    copied_grid[row][col] = 0
                elif copied_grid[row][col] == self.environment.user_space_name:
                    copied_grid[row][col] = 1
                elif copied_grid[row][col] == self.environment.goal_space_name:
                    copied_grid[row][col] = 1
                else:
                    raise ValueError("Invalid values on the grid")
        copied_grid = np.array(copied_grid)
        meta_controller_mask = copied_grid.flatten()

        while not self.episode_over:
            episode_intrinsic_rewards = []
            self.meta_controller_state = self.environment.state
            self.subgoal = self.meta_controller.pick_action(state=self.meta_controller_state, mask = meta_controller_mask)
            self.goals_seen.append(self.subgoal)
            self.subgoal_achieved = False   
            self.cumulative_meta_controller_reward = 0
            
            while not (self.episode_over or self.subgoal_achieved):
                self.state = np.concatenate((self.environment.state, np.array([self.subgoal])))
                self.pick_and_conduct_controller_action(self.subgoal)
                self.update_data()
                if self.time_to_learn(self.controller.memory, self.global_step_number, "CONTROLLER"): #means it is time to train controller
                    for _ in range(self.hyperparameters["CONTROLLER"]["learning_iterations"]):
                        self.controller.learn()
                self.next_state_train = np.concatenate((self.next_state, np.array([self.subgoal])))
                self.save_experience(memory=self.controller.memory, experience=(self.state, self.action, self.reward, self.next_state_train, self.done))
                self.state = self.next_state #this is to set the state for the next iteration
                self.global_step_number += 1
                episode_intrinsic_rewards.append(self.reward)

            if self.time_to_learn(self.meta_controller.memory, self.meta_controller_steps, "META_CONTROLLER"):
                for _ in range(self.hyperparameters["META_CONTROLLER"]["learning_iterations"]):
                    self.meta_controller.learn()

            self.save_experience(memory=self.meta_controller.memory,
                                 experience=(self.meta_controller_state, self.subgoal, self.cumulative_meta_controller_reward,
                                             self.meta_controller_next_state, self.episode_over))
            self.meta_controller_steps += 1
            self.episode_steps += 1

        self.rolling_intrinsic_rewards.append(np.sum(episode_intrinsic_rewards))
        if self.episode_number % 100 == 0:
            print(" ")
            print("Most common goal -- {} -- ".format( max(set(self.goals_seen[-100:]), key=self.goals_seen[-100:].count)  ))
            print("Intrinsic Rewards -- {} -- ".format(np.mean(self.rolling_intrinsic_rewards[-100:])))
            print("Average controller action -- {} ".format(np.mean(self.controller_actions[-100:])))
            print("Latest subgoal -- {}".format(self.goals_seen[-1]))
        self.episode_number += 1
        self.controller.episode_number += 1
        self.meta_controller.episode_number += 1

    def pick_and_conduct_controller_action(self, option = None):
        """Picks and conducts an action for controller"""
        self.action =  self.controller.pick_action(state=self.state)
        self.controller_actions.append(self.action)
        self.conduct_action(self.action, option)

    def update_data(self):
        """Updates stored data for controller and meta-controller. It must occur in the order shown"""
        self.episode_over = self.done
        self.update_controller_data()
        self.update_meta_controller_data()

    def update_controller_data(self):
        """Gets the next state, reward and done information from the environment"""
        environment_next_state = self.next_state
        assert len(environment_next_state) == 2
        # self.next_state = np.concatenate((environment_next_state, np.array([self.subgoal])))
        self.subgoal_achieved = environment_next_state[0] == self.subgoal
        self.reward = 1.0 * self.subgoal_achieved
        self.done = self.subgoal_achieved or self.episode_over

    def update_meta_controller_data(self):
        """Updates data relating to meta controller"""
        self.cumulative_meta_controller_reward += self.reward
        # self.total_episode_score_so_far += self.environment.get_reward()
        if self.done:
            self.meta_controller_next_state = self.next_state

    def time_to_learn(self, memory, steps_taken, controller_name):
        """Boolean indicating whether it is time for meta-controller or controller to learn"""
        enough_experiences = len(memory) > self.hyperparameters[controller_name]["batch_size"]
        enough_steps_taken = steps_taken % self.hyperparameters[controller_name]["update_every_n_steps"] == 0
        return enough_experiences and enough_steps_taken


if __name__== '__main__':
    total_steps = 0
    start_time = time.time()
    possible_next_goals = [74, 75, 84, 85]
    history_steps = np.zeros((config.nruns, config.num_episodes_to_run))
    for run in range(config.nruns):
        env = config.environment
        env.reset()
        env.set_goal(62)

        np.random.seed(config.seed+run)
        random.seed(config.seed+run)

        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(nfeatures, nactions, config.action_temperature) for _ in range(config.noptions)]

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(nfeatures) for _ in range(config.noptions)]

        # Policy over options
        meta_policy = SoftmaxPolicy(nfeatures, config.noptions, config.option_temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(config.discount, config.lr_critic, option_terminations, meta_policy.weights, meta_policy,config.noptions) 

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, config.lr_term,config.noptions)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, config.lr_intra, config.discount, critic, config.noptions)

        tot_steps=0.
        for episode in range(config.num_episodes_to_run):
            if episode > 0 and episode == int(config.num_episodes_to_run/2.): ############################# Change time #############################
                goal = possible_next_goals[config.seed % len(possible_next_goals)]
                env.set_goal(goal)
                print('************* New goal : ', env.current_goal_location)



            
            last_opt=None
            phi = features(env.reset())
            option = meta_policy.sample(phi)
            # print(f" Current option is {option}.")
            action = option_policies[option].sample(phi)
            critic.start(phi, option)



            action_ratios_avg=[]
            
            for step in range(config.nsteps):
                observation, reward, done, _ = env.step(action)
                next_phi = features(features(observation[0]))

                if option_terminations[option].sample(next_phi): 
                    next_option = meta_policy.sample(next_phi)
                else:
                    next_option=option

                next_action = option_policies[next_option].sample(next_phi)



                ###Action ratios
                action_ratios=np.zeros((config.noptions))
                for o in range(config.noptions):
                    action_ratios[o] = option_policies[o].pmf(phi)[0][action]
                action_ratios= action_ratios / action_ratios[option]
                action_ratios_avg.append(action_ratios)


                # Prob of current option
                one_hot = np.zeros(config.noptions)
                if last_opt is not None:
                    bet = option_terminations[last_opt].pmf(phi)
                    one_hot[last_opt] = 1.
                else:
                    bet = 1.0
                prob_curr_opt = bet * meta_policy.pmf(phi) + (1-bet)*one_hot
                one_hot_curr_opt= np.zeros(config.noptions)
                one_hot_curr_opt[option] = 1.
                sampled_eta = float(np.random.rand() < config.eta)
                prob_curr_opt= config.eta * prob_curr_opt + (1 - config.eta) * one_hot_curr_opt


            
                # Critic updates
                critic.update(next_phi, next_option, reward, done, one_hot_curr_opt)


                # Intra-option policy update
                critic_feedback = reward + config.discount * critic.value(next_phi, next_option)
                critic_feedback -= critic.value(phi, option)
                if config.multi_option:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        action_ratios, prob_curr_opt  )   
                else:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        np.ones_like(action_ratios), one_hot_curr_opt  ) 

                # Termination update
                if not done:
                    termination_improvement.update(next_phi, option, one_hot_curr_opt)


                last_opt=option
                phi=next_phi
                option=next_option
                action=next_action


                if done:
                    break


            tot_steps+=step
            history_steps[run, episode] = step
            end=time.time()
            print('Run {} Total steps {} episode {} steps {} FPS {:0.0f} '.format(run,tot_steps, episode, step,   int(tot_steps/ (end- start_time)) )  )



