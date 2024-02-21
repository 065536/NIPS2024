from agents.DQN_agents.DDQN import DDQN
from environments.Four_Rooms_Environment import Four_Rooms_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.hierarchical_agents.h_DQN import h_DQN
from agents.policy_gradient_agents.PPO import PPO

config = Config()
config.seed = 1

height = 15
width = 15
random_goal_place = False
num_possible_states = (height * width) ** (1 + 1*random_goal_place)
embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)

config.environment = Four_Rooms_Environment(height, width, stochastic_actions_probability=0.0, random_start_user_place=True, random_goal_place=random_goal_place)

config.num_episodes_to_run = 1000
config.file_to_save_data_results = "Data_and_Graphs/Four_Rooms.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Four_Rooms.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "DQN_Agents": {
        "linear_hidden_units": [30, 10],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 300,
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [embedding_dimensions[0],
                                     [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False

        }

    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],

        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 50.0,
        "normalise_rewards": True,
        "clip_rewards": False

    },


    "DIAYN": {

        "num_skills": 5,
        "DISCRIMINATOR": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
        },

        "AGENT": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
        }
    },


    "HRL": {
        "linear_hidden_units": [10, 5],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 400,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01

    },

    "h_DQN":{
        "CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1, 2],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau" : 0.01
        },
        "META_CONTROLLER": {
           "batch_size": 32,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 2500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau" : 0.01
        }
    },

    "h_DQN_with_MOC":{
        "CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1, 2],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau" : 0.01
        },
        "META_CONTROLLER": {
            "action_temperature": 1e-2,
            "option_temperature": 1e-1,
            "noptions": config.environment.observation_space.n,
            "discount": 0.99,
            "lr_term": 0.8,
            "lr_intra": 0.8,
            "lr_critic": 0.8,
            "eta": 0.3,
            "multi_option": False,
        }
    }, 

    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "relu",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },
}

if __name__== '__main__':

    # AGENTS = [DDQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    # AGENTS = [h_DQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    AGENTS = [PPO]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


