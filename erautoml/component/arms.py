from erautoml.component.evaluators import evaluate_serial
from erautoml.opt.racos import Dimension, RacosOptimization, SpaceRunOut
import os
import time

import numpy as np
import pandas as pd
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from stable_baselines3.common.callbacks import BaseCallback


def all_dataset_name(path='./data_set'):
    dirs = os.listdir(path)
    return dirs


def racos_hp2search_space(hp_space):

    this_float = 0
    this_integer = 1
    this_categorical = 2

    search_space = []
    for hp in hp_space:
        if hp.is_float_type():
            this_type = this_float
        elif hp.is_int_type():
            this_type = this_integer
        elif hp.is_categorical_type():
            this_type = this_categorical
        else:
            assert False, print("error hyper-parameter type!")
        lower_bound, upper_bound = hp.param_bound
        search_space.append(((lower_bound, upper_bound), this_type))

    return search_space


class Arms:
    def __init__(self, dataset=None, algo_ins=None, metric=None, num_tasks=16, time_to_stop=None, k_fold=5):
        self.dataset = dataset
        self.algo_ins = algo_ins
        self.num_tasks = num_tasks
        self.time_to_stop = time_to_stop
        self.k_fold = k_fold
        self.metric = metric
        self.arm_name = self.algo_ins.__class__.__name__

        self.search_space = racos_hp2search_space(self.algo_ins.get_hp_space())
        self.optimizer = None
        self._init_racos()
        
        return
    
    def _init_racos(self):
        sample_size = 5
        positive_num = 2
        budget = 2000 # may change to a variable
        random_probability = 0.95
        uncertain_bit = int(len(self.search_space) * 0.1) + 1
        dimension = Dimension()
        dimension.set_dimension_size(len(self.search_space))
        for i in range(len(self.search_space)):
            dim_range, dim_type = self.search_space[i]
            dimension.set_region(i, [dim_range[0], dim_range[1]], dim_type)

        self.optimizer = RacosOptimization(dimension)
        self.optimizer.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=random_probability, ub=uncertain_bit)
    
    def sample(self):
        new_x = self.optimizer.sample()

        return new_x

    def update_optimizer(self, x, x_v):

        self.optimizer.update_model(x, x_v)

    def pull(self):
        try:
            hps = self.sample()
        except SpaceRunOut:
            return None
        ret_item = evaluate_serial(self.algo_ins, hps, self.dataset, self.metric, self.k_fold)

        return ret_item


# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq:
#     :param log_dir: Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: Verbosity level.
#     """
#     def __init__(self, check_freq: int, log_dir: str, info: dict, verbose: int = 1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.info = info
#         self.save_path = log_dir
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:

#           # Retrieve training reward
#           x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#           if len(x) > 0:
#               # Mean training reward over the last 100 episodes
#               mean_reward = np.mean(y[-50:])
#               print('mean reward:', mean_reward)
#               if self.verbose > 0:
#                 print(f"Num timesteps: {self.num_timesteps}")
#                 print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

#               # New best model, you could save the agent here
#               if mean_reward > self.best_mean_reward:
#                   self.best_mean_reward = mean_reward
#                   # Example for saving best model
#                   if self.verbose > 0:
#                     print(f"Saving new best model to {self.save_path}")
#                   self.model.save(self.save_path)

#         return True


# def get_score_from_csv(this_dir, window_len=50):
#     filepath = os.path.join(this_dir, 'monitor.csv')
#     df = pd.read_csv(filepath, skiprows=1)
#     all_rew = list(df['r'])
#     length = len(all_rew)
#     if length <= window_len:
#         return np.mean(all_rew)
#     else:
#         max_avg_rew = float('-inf')
#         i = 0
#         j = window_len - 1
#         while j < window_len:
#             max_avg_rew = max(max_avg_rew, np.mean(all_rew[i:j+1]))
#             i += 1
#             j += 1
#         return max_avg_rew

# class RlArms:
#     def __init__(self, algo_ins=None):
#         self.algo_ins = algo_ins
#         self.arm_name = self.algo_ins.__class__.__name__

#         self.search_space = racos_hp2search_space(self.algo_ins.get_hp_space())
#         self.optimizer = None
#         self._init_racos()
        
#         return
    
#     def _init_racos(self):
#         sample_size = 5
#         positive_num = 2
#         budget = 2000 # may change to a variable
#         random_probability = 0.95
#         uncertain_bit = int(len(self.search_space) * 0.1) + 1
#         dimension = Dimension()
#         dimension.set_dimension_size(len(self.search_space))
#         for i in range(len(self.search_space)):
#             dim_range, dim_type = self.search_space[i]
#             dimension.set_region(i, [dim_range[0], dim_range[1]], dim_type)

#         self.optimizer = RacosOptimization(dimension)
#         self.optimizer.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=random_probability, ub=uncertain_bit)
    
#     def sample(self):
#         new_x = self.optimizer.sample()

#         return new_x

#     def update_optimizer(self, x, x_v):

#         self.optimizer.update_model(x, x_v)

#     def pull(self, env=None, timesteps=25000, this_dir=None, check_freq=1000, pull_idx=None):
#         try:
#             hps = self.sample()
#         except SpaceRunOut:
#             return None

#         model = self.algo_ins
#         env = Monitor(env, this_dir)
#         info = {'algo_name': self.algo_ins.algorithm_name, 'pull_idx': pull_idx}
#         model.generate_model(hp_list=hps, env=env, this_dir=this_dir)
#         callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=this_dir, info=info)
#         start = time.time()
#         model.learn(timesteps=timesteps, callback=callback, pull_idx=pull_idx)
#         learn_duration = time.time() - start
#         max_avg_rew = get_score_from_csv(this_dir)

#         return {'pull_idx': pull_idx, 'arm': self.algo_ins.__class__.__name__, 'hps': hps, 'rew': max_avg_rew/100, 'learn_duration': learn_duration}


if __name__ == "__main__":
    pass
