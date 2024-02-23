import copy
import numpy as np
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN
from .utils import *
import torch

class h_DQN_with_MOC(Base_Agent):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat
    Note also that this algorithm only works when we have discrete states and discrete actions currently because otherwise
    it is not clear what it means to achieve a subgoal state designated by the meta-controller"""
    agent_name = "h_DQN_with_MOC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.controller_config = copy.deepcopy(config)
        self.controller_config.output_dim_ = self.action_size
        self.controller_config.magic_number = 3
        self.controller_config.hyperparameters = self.controller_config.hyperparameters["CONTROLLER"]
        self.controller = DDQN(self.controller_config)
        self.controller.q_network_local = self.create_NN(input_dim=self.state_size*3, output_dim=self.action_size,
                                                         key_to_use="CONTROLLER")
        
        self.features = Tabular(self.environment.observation_space.n)
        self.nfeatures, self.nactions = len(self.features), self.environment.action_space.n

        self.meta_controller_config = copy.deepcopy(config)
        self.meta_controller_config.hyperparameters = self.meta_controller_config.hyperparameters["META_CONTROLLER"]
        self.meta_controller_config.output_dim_ = config.environment.observation_space.n
        self.option_policies = [SoftmaxPolicy(self.nfeatures, self.nactions, self.meta_controller_config.hyperparameters["action_temperature"]) for _ in range(self.meta_controller_config.hyperparameters["noptions"])]
        self.option_terminations = [SigmoidTermination(self.nfeatures) for _ in range(self.meta_controller_config.hyperparameters["noptions"])]
        self.meta_policy = SoftmaxPolicy(self.nfeatures, self.meta_controller_config.hyperparameters["noptions"], self.meta_controller_config.hyperparameters["noptions"])
        self.critic = IntraOptionQLearning(self.meta_controller_config.hyperparameters["discount"], self.meta_controller_config.hyperparameters["lr_critic"], self.option_terminations, self.meta_policy.weights, self.meta_policy,self.meta_controller_config.hyperparameters["noptions"]) 
        self.termination_improvement= TerminationGradient(self.option_terminations, self.critic, self.meta_controller_config.hyperparameters["lr_term"], self.meta_controller_config.hyperparameters["noptions"])
        self.intraoption_improvement = IntraOptionGradient(self.option_policies, self.meta_controller_config.hyperparameters["lr_intra"], self.meta_controller_config.hyperparameters["discount"], self.critic, self.meta_controller_config.hyperparameters["noptions"])


        self.rolling_intrinsic_rewards = []
        self.goals_seen = []
        self.controller_learnt_enough = False
        self.controller_actions = []
        self.action_ratios_avg = []

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
        
    
    def get_wall(self):
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
                    copied_grid[row][col] = 0
                else:
                    raise ValueError("Invalid values on the grid")
        copied_grid = np.array(copied_grid)
        meta_controller_mask = copied_grid.flatten()
        return meta_controller_mask

    def step(self):

        self.episode_steps = 0
        self.meta_controller_mask = self.get_wall()

        while not self.episode_over:
            episode_intrinsic_rewards = []
            self.last_option = None
            self.phi = self.features(self.environment.reset())
            self.option = self.meta_policy.sample(self.phi, self.meta_controller_mask)
            self.critic.start(self.phi, self.option)

            self.goals_seen.append(self.option)
            self.subgoal_achieved = False   
            self.cumulative_meta_controller_reward = 0
            
            while not (self.episode_over or self.subgoal_achieved):
                self.state = np.concatenate((self.environment.state, np.array([self.option])))
                self.pick_and_conduct_controller_action(self.option)
                self.update_data()

                self.next_phi = self.features(self.features(self.next_state[0]))

                self.option_critic()

                if self.time_to_learn(self.controller.memory, self.global_step_number, "CONTROLLER"): #means it is time to train controller
                    for _ in range(self.hyperparameters["CONTROLLER"]["learning_iterations"]):
                        self.loss = self.controller.learn()
                self.next_state_train = np.concatenate((self.next_state, np.array([self.option])))
                self.save_experience(memory=self.controller.memory, experience=(self.state, self.action, self.reward, self.next_state_train, self.done))
                self.state = self.next_state #this is to set the state for the next iteration
                self.global_step_number += 1
                episode_intrinsic_rewards.append(self.reward)
 
            self.episode_steps += 1

        self.rolling_intrinsic_rewards.append(np.sum(episode_intrinsic_rewards))
        if self.episode_number % 100 == 0:
            print(" ")
            print("Most common goal -- {} -- ".format( max(set(self.goals_seen[-100:]), key=self.goals_seen[-100:].count)  ))
            print("Intrinsic Rewards -- {} -- ".format(np.mean(self.rolling_intrinsic_rewards[-100:])))
            print("Average controller action -- {} ".format(np.mean(self.controller_actions[-100:])))
            print("Latest subgoal -- {}".format(self.goals_seen[-1]))
            # checkpoint = {
            #     'epoch': self.episode_number,
            #     'loss': loss.item()  
            # }
            # torch.save(checkpoint, f'model_checkpoint_epoch_{self.episode_number}.pt')
        self.episode_number += 1
        self.controller.episode_number += 1

    def pick_and_conduct_controller_action(self, option = None):
        """Picks and conducts an action for controller"""
        self.action =  self.controller.pick_action(state=self.state)
        self.controller_actions.append(self.action)
        self.conduct_action(self.action, option)

    def update_data(self):
        """Updates stored data for controller and meta-controller. It must occur in the order shown"""
        self.episode_over = self.done
        self.update_controller_data()

    def update_controller_data(self):
        """Gets the next state, reward and done information from the environment"""
        environment_next_state = self.next_state
        assert len(environment_next_state) == 2
        # self.next_state = np.concatenate((environment_next_state, np.array([self.subgoal])))
        self.subgoal_achieved = environment_next_state[0] == self.option
        self.reward = 1.0 * self.subgoal_achieved
        self.done = self.subgoal_achieved or self.episode_over


    def time_to_learn(self, memory, steps_taken, controller_name):
        """Boolean indicating whether it is time for meta-controller or controller to learn"""
        enough_experiences = len(memory) > self.hyperparameters[controller_name]["batch_size"]
        enough_steps_taken = steps_taken % self.hyperparameters[controller_name]["update_every_n_steps"] == 0
        return enough_experiences and enough_steps_taken
    
    def option_critic(self):

        if self.option_terminations[self.option].sample(self.next_phi):
            self.next_option = self.meta_policy.sample(self.next_phi, self.meta_controller_mask)
        else:
            self.next_option = self.option

        action_ratios=np.zeros((self.meta_controller_config.hyperparameters["noptions"]))
        for o in range(self.meta_controller_config.hyperparameters["noptions"]):
            action_ratios[o] = self.option_policies[o].pmf(self.phi)[0][self.action]
        action_ratios= action_ratios / action_ratios[self.option]
        self.action_ratios_avg.append(action_ratios)


        # Prob of current option
        one_hot = np.zeros(self.meta_controller_config.hyperparameters["noptions"])
        if self.last_option is not None:
            bet = self.option_terminations[self.last_option].pmf(self.phi)
            one_hot[self.last_option] = 1.
        else:
            bet = 1.0
        prob_curr_opt = bet * self.meta_policy.pmf(self.phi) + (1-bet)*one_hot
        one_hot_curr_opt= np.zeros(self.meta_controller_config.hyperparameters["noptions"])
        one_hot_curr_opt[self.option] = 1.
        sampled_eta = float(np.random.rand() < self.meta_controller_config.hyperparameters["eta"])
        prob_curr_opt= self.meta_controller_config.hyperparameters["eta"] * prob_curr_opt + (1 - self.meta_controller_config.hyperparameters["eta"]) * one_hot_curr_opt


    
        # Critic updates
        self.critic.update(self.next_phi, self.next_option, self.reward, self.done, one_hot_curr_opt)


        # Intra-option policy update
        critic_feedback = self.reward + self.meta_controller_config.hyperparameters["discount"] * self.critic.value(self.next_phi, self.next_option)
        critic_feedback -= self.critic.value(self.phi, self.option)
        if self.meta_controller_config.hyperparameters["multi_option"]:
            self.intraoption_improvement.update(self.phi, self.option, self.action, self.reward, self.done, self.next_phi, self.next_option, critic_feedback,
                action_ratios, prob_curr_opt)   
        else:
            self.intraoption_improvement.update(self.phi, self.option, self.action, self.reward, self.done, self.next_phi, self.next_option, critic_feedback,
                np.ones_like(action_ratios), one_hot_curr_opt  ) 

        # Termination update
        if not self.done:
            self.termination_improvement.update(self.next_phi, self.option, one_hot_curr_opt)
        
        self.last_option = self.option
        self.phi = self.next_phi
        self.option = self.next_option