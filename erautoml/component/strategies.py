from cmath import e
import numpy as np
import heapq 
import math
import random
from erautoml.opt.racos import RacosOptimization, Dimension, SpaceRunOut
import time
from erautoml.component.evaluators import evaluate, Metric, evaluate_serial
import ray

LOG_BASE = 0.05


def threads_allocation(num_threads, sorted_values):
    value_sum = np.sum([x[1] for x in sorted_values])
    wanted_percentage = np.array([x[1]/value_sum for x in sorted_values])
    primary_allocation = [int(x*num_threads) for x in wanted_percentage]
    difference = num_threads - np.sum(primary_allocation)
    actual_percentage = np.array([x/num_threads for x in primary_allocation])
    percentage_difference = wanted_percentage - actual_percentage
    top_k_index = heapq.nlargest(difference, range(len(percentage_difference)), percentage_difference.take)
    for i in top_k_index:
        primary_allocation[i] += 1

    final_allocation = []
    for i, repeat_num in enumerate(primary_allocation):
        final_allocation.append((sorted_values[i][0], repeat_num))

    return final_allocation


def run_strategy_parallel(strategy, total_pulls, time_to_stop, parallel_num, k_fold=5, task='clf'):

    start_time = time.time()
    strategy.reset()
    history = []
    assert strategy.arm_num <= total_pulls, print("total pulls is less than arm number!")

    if isinstance(strategy, ERUCBStrategy):
        
        init_lst = []
        for arm in strategy.arms:
            hps = arm.sample()
            if time.time() > time_to_stop:
                print('==WARNING: Due to time limit, not all arms initialized!')
                break
            init_lst.append(evaluate.remote(arm.algo_ins, hps, arm.dataset, arm.metric, k_fold))
        init_ret = ray.get(init_lst)
        for item in init_ret:
            strategy.idx += 1
            this_history = {'ret_idx': strategy.idx}
            this_history.update(item)
            history.append(this_history)
            arm_name = item['arm']
            arm = strategy.ii[arm_name]
            if arm.metric.larger_better:
                if task=='rgr' and isinstance(strategy, ERUCBStrategy) and LOG_BASE: # update strategy
                    strategy.update(arm_name, math.log(item['score'], LOG_BASE)) 
                else:
                    strategy.update(arm_name, item['score']) 
                arm.update_optimizer(item['hps'], -item['score']) # update racos
            else:
                if task=='rgr' and isinstance(strategy, ERUCBStrategy) and LOG_BASE: # update strategy
                    strategy.update(arm_name, -math.log(item['score'], LOG_BASE)) 
                else:
                    strategy.update(arm_name, -item['score']) 
                arm.update_optimizer(item['hps'], item['score'])

    elif isinstance(strategy, ENUCBStrategy):
        init_lst = []
        for arm in strategy.arms:
            for _ in range(strategy.init_size):
                hps = arm.sample()
                if time.time() > time_to_stop:
                    print('==WARNING: Due to time limit, not all arms initialized!')
                    break
                init_lst.append(evaluate.remote(arm.algo_ins, hps, arm.dataset, arm.metric, k_fold))
        init_ret = ray.get(init_lst)
        for item in init_ret:
            strategy.idx += 1
            this_history = {'ret_idx': strategy.idx}
            this_history.update(item)
            history.append(this_history)
            arm_name = item['arm']
            arm = strategy.ii[arm_name]
            if arm.metric.larger_better:
                feedback = item['score'] # update strategy
                arm.update_optimizer(item['hps'], -item['score']) # update racos
            else:
                feedback = -item['score']
                arm.update_optimizer(item['hps'], item['score'])

            strategy.feedback_list[arm_name].append(feedback)
            strategy.y_list[arm_name].append(strategy._g_inverse(feedback))
            strategy.tn[arm_name] += 1
            strategy.total_t += 1

        for arm in strategy.arms:
            
            arm_name = arm.arm_name
            beta_1, beta_0, sigma = strategy._curve_update(arm_name)
            # if self.print_switch:
            #     print("init {}, curve: {}, z:{}".format(i, (beta_1, beta_0, sigma), self.feedback_list[i]))
            strategy.log.append("init {}, curve: {}, z:{}".format(arm_name, (beta_1, beta_0, sigma), strategy.feedback_list[arm_name]))
            strategy.curve_param_list[arm_name] = (beta_1, beta_0, sigma)


    strategy.init_history = history.copy()
    
    remaining_pulls = total_pulls - strategy.arm_num
    sorted_values = strategy.refresh_decision_values()
    strategy.decision_values_history['init_phase'] = strategy.decision_values
    total_threads = min(remaining_pulls, parallel_num)

    strategy.threads_allocation = threads_allocation(total_threads, sorted_values)

    ray_list = []
    
    waiting_ids = []
    for name, repeat_num in strategy.threads_allocation:
        arm = strategy.ii[name]
        for i in range(repeat_num):
            try:
                hps = arm.sample()
                waiting_ids.append(evaluate.remote(arm.algo_ins, hps, arm.dataset, arm.metric, k_fold))
                ray_list.append(name)
            except SpaceRunOut:
                print('ERROR: ', name, 'space run out during initializing, please make the space larger.')

    remaining_pulls = remaining_pulls - parallel_num
    while len(waiting_ids) > 0: # Loop until all tasks have completed
        ready_ids, remaining_ids = ray.wait(waiting_ids, num_returns=1)
        item = ray.get(ready_ids)[0]

        strategy.idx += 1
        this_history = {'ret_idx': strategy.idx}
        this_history.update(item)
        history.append(this_history)
        arm_name = item['arm']
        arm = strategy.ii[arm_name]
        ray_list.remove(arm_name)
        
        if arm.metric.larger_better:
            if task=='rgr' and isinstance(strategy, ERUCBStrategy) and LOG_BASE: # update strategy
                strategy.update(arm_name, math.log(item['score'], LOG_BASE)) 
            else:
                strategy.update(arm_name, item['score']) 
            arm.update_optimizer(item['hps'], -item['score']) # update racos
        else:
            if task=='rgr' and isinstance(strategy, ERUCBStrategy) and LOG_BASE:
                strategy.update(arm_name, -math.log(item['score'], LOG_BASE))
            else:
                strategy.update(arm_name, -item['score'])
            arm.update_optimizer(item['hps'], item['score'])

        if time.time() > time_to_stop:
            print('===time out')
            for one in remaining_ids:
                ray.cancel(one)
            break

        if remaining_pulls > 0:
            while True:
                if len(strategy.arms) <= 0:
                    print('EARLY STOPPED: no remaining arms.')
                    return history
                name = strategy.select()
                arm = strategy.ii[name]
                try:
                    hps = arm.sample()
                    strategy.decision_values_history['select_phase'].append(strategy.decision_values)
                    break
                except SpaceRunOut:
                    decision_values = strategy.decision_values
                    decision_values.append('not evaluated due to arm delete')
                    strategy.decision_values_history['select_phase'].append(decision_values)
                    print('*-*-delete arm', arm.arm_name)
                    strategy.arms.remove(arm)
            id_to_append = evaluate.remote(arm.algo_ins, hps, arm.dataset, arm.metric, k_fold)
            ray_list.append(name)
            remaining_ids.append(id_to_append)
            remaining_pulls -= 1
        waiting_ids = remaining_ids  # Reset this list; don't include the completed ids in the list again!
    
    strategy.total_eval_time = time.time() - start_time

    return history


class BaseStrategy(object):

    def __init__(self, arms=None):
        self.log = []
        self.arm_num = len(arms)
        self.arms = arms
        self.original_arms = arms
        self.ii = {}
        self.total_eval_time = 0
        for arm in self.original_arms:
            self.ii[arm.arm_name] = arm

        return

    def refresh_ii(self):
        self.ii = {}
        for arm in self.original_arms:
            self.ii[arm.arm_name] = arm

        return

    def reset(self):
        return

    def select(self):
        return

    def update(self, arm_index, feedback):
        return

    def run_strategy(self, total_tries, arms):
        return


class ERUCBStrategy(BaseStrategy):

    def __init__(self, arms=None, theta=0.1, beta=0.5, gamma=1.0):
        super(ERUCBStrategy, self).__init__(arms=arms)
        # hyper-parameter initialization of ER-UCB
        self.theta = theta
        self.beta = beta
        self.gamma = gamma

        self.tn = {}
        self.mu_y = {}
        self.mu_z = {}
        self.feedback_list = {}

        self.idx = 0
        self.init_history = []
        self.decision_values = []
        self.decision_values_history = {'init_phase': None, 'select_phase': []}
        self.sorted_decision_values = []
        self.threads_allocation = []
        self.reset()

        return

    # whether initialization step is finished, if it is finished, return true
    def is_lawful_select(self):
        for arm in self.arms:
            if arm.arm_name in self.tn:
                if self.tn[arm.arm_name] == 0:
                    return False
            else:
                return False
        return True


    def reset(self):
        for arm in self.arms:
            self.tn[arm.arm_name] = 0
            self.mu_y[arm.arm_name] = 0.0
            self.mu_z[arm.arm_name] = 0.0
            self.feedback_list[arm.arm_name] = []
        self.total_t = 0
        self.idx = 0
        self.init_history = []
        self.decision_values = []
        self.decision_values_history = {'init_phase': None, 'select_phase': []}
        self.sorted_decision_values = []
        self.threads_allocation = []
        
        return

    def drop_arm(self, arm_name):
        self.mu_y.pop(arm_name)
        self.mu_z.pop(arm_name)
        self.arms.remove(self.ii[arm_name])
        return

    def refresh_decision_values(self):
        self.decision_values = []
        for arm in self.arms:
            # calculate score of each arm
            this_arm_score = self.gamma * (self.mu_y[arm.arm_name] + math.sqrt(self.mu_z[arm.arm_name] / self.theta))\
                             + math.sqrt(2.0 * math.log(self.total_t) / self.tn[arm.arm_name])\
                             + math.sqrt(math.sqrt(2.0 * math.log(self.total_t) / self.tn[arm.arm_name]) / self.theta)
            if math.isnan(this_arm_score):
                this_arm_score = 1.5
                print('score not found, default given', arm.arm_name, this_arm_score)
            self.decision_values.append((arm.arm_name, this_arm_score))
        self.sorted_decision_values = sorted(self.decision_values, key=lambda x: x[1], reverse=True)

        return self.sorted_decision_values
    
    # select an arm from candidates
    def select(self):
        if not self.is_lawful_select():
            for arm in self.arms:
                if arm.arm_name not in self.tn or self.tn[arm.arm_name]==0:
                    return arm.arm_name
        self.refresh_decision_values()
        return self.sorted_decision_values[0][0]

    # update the arm according to the feedback
    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        self.mu_y[arm_name] = (self.tn[arm_name] * self.mu_y[arm_name] + feedback - self.beta)\
                               / (self.tn[arm_name] + 1)
        self.mu_z[arm_name] = (self.tn[arm_name] * self.mu_z[arm_name] + pow((feedback - self.beta), 2))\
                               / (self.tn[arm_name] + 1)
        self.tn[arm_name] += 1
        self.total_t += 1
        return


class ENUCBStrategy(BaseStrategy):

    def __init__(self, arms=None, t_axis_scale=0.025, alpha=4.0, theta=0.01, gamma=1.0, init_size=2, g_name="sigmoid",
                 print_switch=False):
        super(ENUCBStrategy, self).__init__(arms=arms)
        self.arm_num = len(self.arms)
        self.print_switch = print_switch
        self.log.append("strategy: {} ---------------------------".format("en-ucb"))
        if self.print_switch:
            print("strategy: {} ---------------------------".format("en-ucb"))
        self.t_axis_scale = t_axis_scale
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.init_size = init_size
        self.g_name = g_name
        self.log.append("hyper-parameters: --")
        self.log.append("   t axis scale: ".format(self.t_axis_scale))
        self.log.append("   alpha: {}, theta: {}, gamma: {}".format(self.alpha, self.theta, self.gamma))
        self.log.append("   init size: {}, g_name: {}".format(self.init_size, self.g_name))
        if self.print_switch:
            print("hyper-parameters: --")
            print("t axis scale: ".format(self.t_axis_scale))
            print("alpha: {}, theta: {}, gamma: {}".format(self.alpha, self.theta, self.gamma))
            print("init size: {}".format(self.init_size))
        # the parameters (beta_1, beta_0, sigma) of convergence curve
        self.curve_param_list = {}
        self.tn = {}
        self.y_list = {}
        self.total_tries = 0 # total try budget in the running of strategy
        self.curve_log = []
        self.feedback_list = {}
        self.init_idx = -1
        self.idx = 0
        self.decision_values_history = {'init_phase': None, 'select_phase': []}
        self.reset()
        return

    def reset(self):
        for arm in self.arms:
            self.y_list[arm.arm_name] = []
            self.feedback_list[arm.arm_name] = []
            self.tn[arm.arm_name] = 0
        self.total_t = 0
        self.curve_param_list = {}
        self.curve_log = []
        self.decision_values_history = {'init_phase': None, 'select_phase': []}
        self.idx = 0

        return

    def drop_arm(self, arm_name):
        self.y_list.pop(arm_name)
        self.arms.remove(self.ii[arm_name])
        return

    # y = g^{-1}(x)
    def _g_inverse(self, x):
        if self.g_name == "linear":
            y = x
        elif self.g_name == "ln":
            y = math.exp(x) - 1
        elif self.g_name == "sigmoid":
            if x > 1:
                print("error x: {}".format(x))
            y = math.log((1 + x) / abs(1 - x)) / self.t_axis_scale
        else:
            y = 0
        return y

    # z = g(y)
    def _g(self, y):

        if self.g_name == "linear":
            z = y
        elif self.g_name == "ln":
            z = math.log(y + 1)
        elif self.g_name == "sigmoid":
            z = 2.0 / (1 + math.exp(-self.t_axis_scale * y)) - 1
        else:
            z = 0
        return z

    # linear regression based on ys
    def _curve_update(self, arm_name):

        assert self.tn[arm_name] == len(self.y_list[arm_name]), print("enucb curve update error!")
        this_ys = np.array(self.y_list[arm_name])
        this_zs = np.array(self.feedback_list[arm_name])
        this_ts = np.array([_ + 1.0 for _ in range(len(self.y_list[arm_name]))])
        AT = np.dot(this_ts - np.mean(this_ts), (this_ts - np.mean(this_ts)).T)
        # AT2 = (np.max(this_ts) ** 3 - np.max(this_ts)) / 12.0
        beta_1 = np.sum((this_ts - np.mean(this_ts)) * this_ys) / AT
        beta_0 = np.mean(this_ys) - beta_1 * np.mean(this_ts)
        cal_ys = np.array([self._g(beta_1 * this_ts[_] + beta_0) for _ in range(this_ts.shape[0])])
        sigma = math.sqrt(np.mean((this_zs - cal_ys) * (this_zs - cal_ys)))

        return beta_1, beta_0, sigma

    # Delta_{T}(x)
    def _func_delta_t(self, arm_name, x):

        beta_1, beta_0, sigma = self.curve_param_list[arm_name]
        this_ts = np.array([_ + 1.0 for _ in range(len(self.y_list[arm_name]))])
        AT = np.dot(this_ts - np.mean(this_ts), (this_ts - np.mean(this_ts)).T)
        # part_1 = -(x * sigma / math.sqrt(AT)) * norm.ppf(abs(math.pow(self.total_t, -self.alpha)
        #                                                      - self.A_m * math.pow(self.tn[arm_index], -0.5)))
        # part_2 = -sigma * math.sqrt(1.0 / self.tn[arm_index] + math.pow(np.mean(this_ts), 2) / AT)\
        #          * norm.ppf(abs(math.pow(self.total_t, -self.alpha) - self.A_n * math.pow(self.tn[arm_index], -0.5)))
        part_1 = x * sigma / math.sqrt(AT)
        part_2 = sigma * math.sqrt(1.0 / self.tn[arm_name] + math.pow(np.mean(this_ts), 2) / AT)
        r_value = part_1 + part_2

        return r_value

    def _get_score(self, arm_name, t):

        beta_1, beta_0, sigma = self.curve_param_list[arm_name]

        exploitation_item = self._g(beta_1 * t + beta_0) + math.sqrt((sigma ** 2)/self.theta)
        exploration_item = self._func_delta_t(arm_name, self.total_t) +\
                           math.sqrt((self._func_delta_t(arm_name, self.tn[arm_name]) + 1)
                                     * math.sqrt(self.alpha * math.log(self.total_t) / (2 * self.tn[arm_name])))
        this_score = self.gamma * exploitation_item + exploration_item
        # if self.print_switch:
        #     print("score: {}, exploit: {}, explore: {}, t: {}, T: {}".format(this_score, exploitation_item,
        #                                                                      exploration_item, t, self.tn[arm_index]))
        return this_score

    def _initialize_curve(self, arms):
        
        assert len(arms) == self.arm_num, print("initialize curve error!")

        # if self.print_switch:
        #     print("start initialize curve step ---")

        for arm_name in self.arms:
            for j in range(self.init_size):
                feedback = arms[arm_name].pull()
                self.feedback_list[arm_name].append(feedback)
                self.feedback_list[arm_name].append(feedback)
                self.y_list[arm_name].append(self._g_inverse(feedback))
                self.tn[arm_name] += 1
                self.total_t += 1
                if self.print_switch:
                    print("en-ucb run {}: {} - {}".format(arm_name*self.init_size+j, arm_name, feedback))
            beta_1, beta_0, sigma = self._curve_update(arm_name)
            # if self.print_switch:
            #     print("init {}, curve: {}, z:{}".format(i, (beta_1, beta_0, sigma), self.feedback_list[i]))
            self.log.append("init {}, curve: {}, z:{}".format(arm_name, (beta_1, beta_0, sigma), self.feedback_list[arm_name]))
            self.curve_param_list[arm_name] = (beta_1, beta_0, sigma)
        if self.print_switch:
            print("end initialize --")

        return
        

    def refresh_decision_values(self):
        # if self.print_switch:
        #     print("get score -----")
        self.decision_values = []
        for arm in self.arms:
            score = self._get_score(arm.arm_name, self.total_tries)
            if math.isnan(score):
                score = 1.5
                print('score nan, default given', arm.arm_name, score)
            self.decision_values.append((arm.arm_name, score))
        # if self.print_switch:
        #     print("all score -----")
        #     print("{}".format(decision_value_list))
        self.sorted_decision_values = sorted(self.decision_values, key=lambda x: x[1], reverse=True)

        return self.sorted_decision_values

    def is_lawful_select(self):
        return self.total_t >= self.init_size * len(self.arms)

    def select(self):
        if not self.is_lawful_select():
            self.init_idx += 1
            if self.init_idx >= len(self.arms):
                self.init_idx = 0
            return self.arms[self.init_idx].arm_name
        self.refresh_decision_values()
        return self.sorted_decision_values[0][0] # arm name

    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        self.y_list[arm_name].append(self._g_inverse(feedback))
        self.tn[arm_name] += 1
        self.total_t += 1
        beta_1, beta_0, sigma = self._curve_update(arm_name)
        self.curve_param_list[arm_name] = (beta_1, beta_0, sigma)
        return

    def run_strategy(self, total_tries, arms):

        assert arms is not None, print("strategy or arms is None!")
        assert self.arm_num == len(arms), print("strategy is not suitable for the arm number!")
        assert self.arm_num <= total_tries, print("total number of trying is less than arm number!")
        assert self.arm_num * self.init_size, print("total number is too small!")

        self.total_tries = total_tries

        self.feedback_list = []

        self._initialize_curve(arms)
        self.curve_log.append("{}: {}".format(self.init_size - 1, self.curve_param_list))

        for t in range(self.init_size * self.arm_num, total_tries):
            # if self.print_switch:
            #     print("--curve {}: {}--".format(t, self.curve_param_list))
            selected_index = self.select()
            feedback = arms[selected_index].pull()
            self.feedback_list.append(feedback)
            self.update(selected_index, feedback)
            self.log.append("arm {}, z: {}, curve: {}".format(selected_index, feedback,
                                                              self.curve_param_list[selected_index]))
            if self.print_switch:
                print("en-ucb run {}: {} - {}".format(t, selected_index, feedback))
            # if self.print_switch:
            #     print("arm {}, z: {}, curve: {}".format(selected_index, feedback, self.curve_param_list[selected_index]))
            self.curve_log.append("{} : {}".format(t, self.curve_param_list))

        self.log.append("curve change: ---")
        self.log.extend(self.curve_log)

        return self.feedback_list

class ClassicUCBStrategy(BaseStrategy):

    def __init__(self, arms=None):
        super(ClassicUCBStrategy, self).__init__(arms=arms)
        self.tn = {}
        self.mu = {}
        self.feedback_list = {}
        self.reset()
        
        return

    def is_lawful_select(self):
        for arm in self.arms:
            if arm.arm_name in self.tn:
                if self.tn[arm.arm_name] == 0:
                    return False
            else:
                return False
        return True


    def reset(self):
        for arm in self.arms:
            self.tn[arm.arm_name] = 0
            self.mu[arm.arm_name] = 0.0
            self.feedback_list[arm.arm_name] = []
        self.total_t = 0
        self.decision_values = []
        self.sorted_decision_values = []

        return

    def drop_arm(self, arm_name):
        self.mu.pop(arm_name)
        self.arms.remove(self.ii[arm_name])
        return

    def select(self):
        if not self.is_lawful_select():
            for arm in self.arms:
                if arm.arm_name not in self.tn or self.tn[arm.arm_name]==0:
                    return arm.arm_name

        self.decision_values = []
        for arm in self.arms:
            # calculate score of each arm
            this_arm_score = self.mu[arm.arm_name] + math.sqrt(2.0 * math.log(self.total_t) / self.tn[arm.arm_name])
            self.decision_values.append((arm.arm_name, this_arm_score))

        self.sorted_decision_values = sorted(self.decision_values, key=lambda x: x[1], reverse=True)
        return self.sorted_decision_values[0][0]

    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        self.mu[arm_name] = (self.tn[arm_name] * self.mu[arm_name] + feedback) / (self.tn[arm_name] + 1)
        self.tn[arm_name] += 1
        self.total_t += 1
        return


class RandomStrategy(BaseStrategy):

    def __init__(self, arms):
        super(RandomStrategy, self).__init__(arms=arms)
        self.feedback_list = {}
        self.reset()
        return

    def reset(self):
        for arm in self.arms:
            self.feedback_list[arm.arm_name] = []
        return

    def drop_arm(self, arm_name):
        self.arms.remove(self.ii[arm_name])
        return

    def select(self):
        arm = random.choice(self.arms)
        return arm.arm_name

    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        return


class EpsilonGreedyStrategy(BaseStrategy):

    def __init__(self, arms=None, epsilon=0.1):
        super(EpsilonGreedyStrategy, self).__init__(arms=arms)
        self.epsilon = epsilon
        self.feedback_list = {}
        self.tn = {}
        self.mu = {}
        self.reset()
        return

    def reset(self):
        for arm in self.arms:
            self.tn[arm.arm_name] = 0
            self.mu[arm.arm_name] = 0.0
            self.feedback_list[arm.arm_name] = []

        return

    def drop_arm(self, arm_name):
        self.mu.pop(arm_name)
        self.arms.remove(self.ii[arm_name])
        return

    def select(self):
        prob = random.random()
        if prob < self.epsilon:
            arm = random.choice(self.arms)
            return arm.arm_name
        else:
            return max(self.mu, key=lambda x: self.mu[x])

    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        self.mu[arm_name] = (self.mu[arm_name] * self.tn[arm_name] + feedback) / (self.tn[arm_name] + 1)
        self.tn[arm_name] += 1
        return


class SoftmaxStrategy(BaseStrategy):

    def __init__(self, arms, tau=1.0):
        super(SoftmaxStrategy, self).__init__(arms=arms)
        # tau is a hyper-parameter of softmax strategy
        self.tau = tau
        self.feedback_list = {}
        self.tn = {}
        self.mu = {}
        self.reset()
        return

    def reset(self):
        for arm in self.arms:
            self.tn[arm.arm_name] = 0
            self.mu[arm.arm_name] = 0.0
            self.feedback_list[arm.arm_name] = []

        return
        
    def drop_arm(self, arm_name):
        self.mu.pop(arm_name)
        self.arms.remove(self.ii[arm_name])
        return

    def select(self):
        sum_q = 0
        Qs_dict = {}
        for arm in self.arms:
            value = pow(math.e, self.mu[arm.arm_name]/self.tau)
            sum_q += value
            Qs_dict[arm.arm_name] = value
        probs = {}
        for arm in self.arms:
            probs[arm.arm_name] = Qs_dict[arm.arm_name] / sum_q
        sum_prob = 0.0
        prob = random.random()
        for arm in self.arms:
            sum_prob += probs[arm.arm_name]
            if sum_prob >= prob:
                return arm.arm_name
        return self.arms[-1].arm_name

    def update(self, arm_name, feedback):
        self.feedback_list[arm_name].append(feedback)
        self.mu[arm_name] = (self.mu[arm_name] * self.tn[arm_name] + feedback) / (self.tn[arm_name] + 1)
        self.tn[arm_name] += 1
        return


if __name__ == "__main__":
    print("hello world!")