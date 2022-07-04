"""
RACOS is an algorithm for solving derivative-free non-convex optimization problems
In this class, there are three RACOS methods aimed for continuous, discrete and mixed optimization

Author:
    Yi-Qi Hu

Time:
    2016.6.13
"""

'''
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 Copyright (C) 2015 Nanjing University, Nanjing, China
'''

import random
import numpy as np
import math


class SpaceRunOut(Exception):

    def __init__(self):
        return

    def __str__(self):
        print("Racos search space run out.")


def time_formulate(start_t, end_t):
    time_l = end_t - start_t
    if time_l < 0:
        print('time error!')
        hour = 0
        minute = 0
        second = 0
    else:
        hour = int(time_l / 3600)
        time_l = time_l - hour * 3600
        minute = int(time_l / 60)
        second = time_l - minute * 60
    return hour, minute, second


# Instance class, each sample is an instance
class Instance:
    __feature = []  # feature value in each dimension
    __fitness = 0  # fitness of objective function under those features

    def __init__(self, dim):
        self.__feature = []
        for i in range(dim.get_size()):
            self.__feature.append(0)
        self.__fitness = 0
        self.__dimension = dim

    # return feature value in index-th dimension
    def get_feature(self, index):
        return self.__feature[index]

    # return features of all dimensions
    def get_features(self):
        return self.__feature

    # set feature value in index-th dimension
    def set_feature(self, index, v):
        self.__feature[index] = v

    # set features of all dimension
    def set_features(self, v):
        self.__feature = v

    # return fitness under those features
    def get_fitness(self):
        return self.__fitness

    # set fitness
    def set_fitness(self, fit):
        self.__fitness = fit

    #
    def equal(self, ins):
        if len(self.__feature) != len(ins.__feature):
            return False
        for i in range(len(self.__feature)):
            if not math.isclose(self.__feature[i], ins.__feature[i]):
                return False
        return True

    # copy this instance
    def copy_instance(self):
        copy = Instance(self.__dimension)
        for i in range(len(self.__feature)):
            copy.set_feature(i, self.__feature[i])
        copy.set_fitness(self.__fitness)
        return copy

    def show_instance(self):
        print('func-v:', self.__fitness, ' - ', self.__feature)


# Dimension class
# dimension message
class Dimension:
    __size = 0  # dimension size
    __region = []  # lower and upper bound in each dimension
    __type = []  # the type of each dimension, 0 is float, 1 is integer, 2 is categorical

    def __init__(self):
        return

    def set_dimension_size(self, s):
        self.__size = s
        self.__region = []
        self.__type = []

        for i in range(s):
            ori_reg = [0, 0]
            self.__region.append(ori_reg)
            self.__type.append(0)
        return

    # set index-th dimension region
    def set_region(self, index, reg, ty):
        self.__region[index][0] = reg[0]
        self.__region[index][1] = reg[1]
        self.__type[index] = ty
        return

    def set_regions(self, regs, tys):
        for i in range(self.__size):
            self.__region[i][0] = regs[i][0]
            self.__region[i][1] = regs[i][1]
            self.__type[i] = tys[i]
        return

    def get_size(self):
        return self.__size

    def get_region(self, index):
        return self.__region[index]

    def get_regions(self):
        return self.__region

    def get_type(self, index):
        return self.__type[index]

    def is_finite(self):
        for i in range(self.__size):
            # float is marked as 0
            if self.get_type(i) == 0:
                return False
        return True

    def get_space_capacity(self):
        if self.is_finite():
            capacity = 1
            for i in range(self.__size):
                capacity *= (self.get_region(i)[1] - self.get_region(i)[0] + 1)
            return capacity
        else:
            return 'infinite'

# used for generating number randomly
class RandomOperator:

    def __init__(self):
        return

    def get_uniform_integer(self, lower, upper):
        return random.randint(lower, upper)

    def get_uniform_double(self, lower, upper):
        return random.uniform(lower, upper)


class RacosOptimization:

    def __init__(self, dimension):

        self.__pop = []  # population set
        self.__pos_pop = []  # positive sample set
        self.__optimal = None  # the best sample so far
        self.__region = []  # the region of model
        self.__label = []  # the random label, if true random in this dimension
        self.__sample_size = 0  # the instance size of sampling in an iteration
        self.__budget = 0  # budget of evaluation
        self.__positive_num = 0  # positive sample set size
        self.__rand_probability = 0.0  # the probability of sampling in model
        self.__uncertain_bit = 0  # the dimension size of sampling randomly
        self.__dimension = dimension
        self.__dim_capacity = self.__dimension.get_space_capacity()

        self.__sample_count = 0
        self.__generated_sample_count = 0

        # sampling saving
        self.__model_ins = []  # positive sample used to modeling
        self.__negative_set = []  # negative set used to modeling
        self.__new_ins = []  # new sample
        self.__sample_label = []  # the label of each sample

        self.__generated = [] # all generated samples
        self.__generated_features = []
        self.__all_cbnt_of_finite_space = []
        self.__reach_finite_threshhold = False

        for i in range(self.__dimension.get_size()):
            region = [0.0, 0.0]
            self.__region.append(region)
            self.__label.append(0)

        self.__ro = RandomOperator()
        return

    # --------------------------------------------------
    # debugging
    # print positive set
    def show_pos_pop(self):
        print('positive set:------------------')
        for i in range(self.__positive_num):
            self.__pos_pop[i].show_instance()
        print('-------------------------------')
        return

    # print negative set
    def show_pop(self):
        print('negative set:------------------')
        for i in range(self.__sample_size):
            self.__pop[i].show_instance()
        print('-------------------------------')
        return

    # print optimal
    def show_optimal(self):
        print('optimal:-----------------------')
        self.__optimal.show_instance()
        print('-------------------------------')

    # print region
    def show_region(self):
        print('region:------------------------')
        for i in range(self.__dimension.get_size()):
            print('dimension ', i, ' [', self.__region[i][0], ',', self.__region[i][1], ']')
        print('-------------------------------')
        return

    # print label
    def show_label(self):
        print('label:-------------------------')
        print(self.__label)
        print('-------------------------------')

    # --------------------------------------------------

    # clear sampling log
    def log_clear(self):
        self.__model_ins = []
        self.__negative_set = []
        self.__new_ins = []
        self.__sample_label = []
        return

    # get log
    def get_log(self):
        return self.__model_ins, self.__negative_set, self.__new_ins, self.__sample_label

    # generate environment that should be logged
    # return is 2-d list
    def generate_environment(self, ins, new_ins):

        trajectory = []
        if True:
            array_best = []
            for i in range(len(ins.get_features())):
                array_best.append(ins.get_features()[i])
            new_sam = []
            for i in range(len(new_ins.get_features())):
                new_sam.append(new_ins.get_features()[i])
            sorted_neg = sorted(self.__pop, key=lambda instance: instance.get_fitness())
            for i in range(self.__sample_size):
                trajectory.append(sorted_neg[i].get_features())

        return array_best, trajectory, new_sam

    # clear parameters of saving
    def clear(self):
        self.__pop = []
        self.__pos_pop = []
        self.__optimal = None
        self.__sample_count = 0
        self.__generated_sample_count = 0
        return

    # parameters setting
    def set_parameters(self, ss=0, bud=0, pn=0, rp=0.0, ub=0):
        self.__sample_size = ss
        self.__budget = bud
        self.__positive_num = pn
        self.__rand_probability = rp
        self.__uncertain_bit = ub
        return

    # get optimal
    def get_optimal(self):
        return self.__optimal

    # generate an instance randomly
    def random_instance(self, dim, region, label):
        ins = Instance(dim)
        for i in range(dim.get_size()):
            if label[i] is True:
                if dim.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
        return ins

    # generate an instance based on a positive sample
    def pos_random_instance(self, dim, region, label, pos_instance):
        ins = Instance(dim)
        for i in range(dim.get_size()):
            if label[i] is False:
                if dim.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
            else:
                ins.set_feature(i, pos_instance.get_feature(i))
        return ins

    # reset model
    def reset_model(self):
        for i in range(self.__dimension.get_size()):
            self.__region[i][0] = self.__dimension.get_region(i)[0]
            self.__region[i][1] = self.__dimension.get_region(i)[1]
            self.__label[i] = True
        return

    # if an instance exists in a list, return true
    def instance_in_list(self, ins, this_list, end):
        for i in range(len(this_list)):
            if i == end:
                break
            if ins.equal(this_list[i]) is True:
                return True
        return False

    # initialize pop, pos_pop, optimal
    def initialize(self, func):

        temp = []

        self.reset_model()

        # sample in original region under uniform distribution
        for i in range(self.__sample_size + self.__positive_num):
            while True:
                ins = self.random_instance(self.__dimension, self.__region, self.__label)
                if self.instance_in_list(ins, temp, i) is False:
                    break
            ins.set_fitness(func(ins.get_features()))
            temp.append(ins)

        # sorted by fitness
        temp.sort(key=lambda instance: instance.get_fitness())

        # initialize pos_pop
        for i in range(self.__positive_num):
            self.__pos_pop.append(temp[i])

        # initialize pop
        for i in range(self.__sample_size):
            self.__pop.append(temp[self.__positive_num + i])

        # initialize optimal
        self.__optimal = self.__pos_pop[0].copy_instance()

        return

    # distinguish function for mixed optimization
    def distinguish(self, exa, neg_set):
        for i in range(self.__sample_size):
            j = 0
            while j < self.__dimension.get_size():
                if self.__dimension.get_type(j) == 0 or self.__dimension.get_type(j) == 1:
                    if neg_set[i].get_feature(j) < self.__region[j][0] or neg_set[i].get_feature(j) > self.__region[j][
                        1]:
                        break
                else:
                    if self.__label[j] is False and exa.get_feature(j) != neg_set[i].get_feature(j):
                        break
                j += 1
            if j >= self.__dimension.get_size():
                return False
        return True

    # update positive set and negative set using a new sample by online strategy
    def online_update(self, ins):

        # update positive set
        j = 0
        while j < self.__positive_num:
            if ins.get_fitness() < self.__pos_pop[j].get_fitness():
                break
            else:
                j += 1

        if j < self.__positive_num:
            temp = ins
            ins = self.__pos_pop[self.__positive_num - 1]
            k = self.__positive_num - 1
            while k > j:
                self.__pos_pop[k] = self.__pos_pop[k - 1]
                k -= 1
            self.__pos_pop[j] = temp

        # update negative set
        j = 0
        while j < self.__sample_size:
            if ins.get_fitness() < self.__pop[j].get_fitness():
                break
            else:
                j += 1
        if j < self.__sample_size:
            temp = ins
            ins = self.__pop[self.__sample_size - 1]
            k = self.__sample_size - 1
            while k < j:
                self.__pop[k] = self.__pop[k - 1]
            self.__pop[j] = temp

        return

    # update optimal
    def update_optimal(self):
        if self.__pos_pop[0].get_fitness() < self.__optimal.get_fitness():
            self.__optimal = self.__pos_pop[0].copy_instance()
        return

    # generate instance randomly based on positive sample for mixed optimization
    def pos_random_mix_isntance(self, exa, region, label):
        ins = Instance(self.__dimension)
        for i in range(self.__dimension.get_size()):
            if label[i] is False:
                ins.set_feature(i, exa.get_feature(i))
            else:
                # float random
                if self.__dimension.get_type(i) == 0:
                    ins.set_feature(i, self.__ro.get_uniform_double(region[i][0], region[i][1]))
                # integer random
                elif self.__dimension.get_type(i) == 1:
                    ins.set_feature(i, self.__ro.get_uniform_integer(region[i][0], region[i][1]))
                # categorical random
                else:
                    ins.set_feature(i, self.__ro.get_uniform_integer(self.__dimension.get_region(i)[0],
                                                                     self.__dimension.get_region(i)[1]))
        return ins

    # generate model based on mixed optimization for next sampling
    # label[i] = false means this dimension should be set as the value in same dimension of positive sample
    def shrink_model(self, exa, neg_set):

        dist_count = 0

        chosen_dim = []
        non_chosen_dim = [i for i in range(self.__dimension.get_size())]

        remain_neg = [i for i in range(self.__sample_size)]

        while len(remain_neg) != 0:

            dist_count += 1

            temp_dim = non_chosen_dim[self.__ro.get_uniform_integer(0, len(non_chosen_dim) - 1)]
            chosen_neg = self.__ro.get_uniform_integer(0, len(remain_neg) - 1)
            # float dimension shrink
            if self.__dimension.get_type(temp_dim) == 0:
                if exa.get_feature(temp_dim) < neg_set[remain_neg[chosen_neg]].get_feature(temp_dim):
                    temp_v = self.__ro.get_uniform_double(exa.get_feature(temp_dim),
                                                          neg_set[remain_neg[chosen_neg]].get_feature(temp_dim))
                    if temp_v < self.__region[temp_dim][1]:
                        self.__region[temp_dim][1] = temp_v
                else:
                    temp_v = self.__ro.get_uniform_double(neg_set[remain_neg[chosen_neg]].get_feature(temp_dim),
                                                          exa.get_feature(temp_dim))
                    if temp_v > self.__region[temp_dim][0]:
                        self.__region[temp_dim][0] = temp_v
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) < self.__region[temp_dim][0] or \
                            neg_set[remain_neg[r_i]].get_feature(temp_dim) > self.__region[temp_dim][1]:
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1
            # integer dimension shrink
            elif self.__dimension.get_type(temp_dim) == 1:
                if exa.get_feature(temp_dim) < neg_set[remain_neg[chosen_neg]].get_feature(temp_dim):
                    temp_v = self.__ro.get_uniform_integer(exa.get_feature(temp_dim),
                                                           neg_set[remain_neg[chosen_neg]].get_feature(temp_dim) - 1)
                    if temp_v < self.__region[temp_dim][1]:
                        self.__region[temp_dim][1] = temp_v
                else:
                    temp_v = self.__ro.get_uniform_integer(neg_set[remain_neg[chosen_neg]].get_feature(temp_dim) - 1,
                                                           exa.get_feature(temp_dim))
                    if temp_v > self.__region[temp_dim][0]:
                        self.__region[temp_dim][0] = temp_v
                if self.__region[temp_dim][0] == self.__region[temp_dim][1]:
                    chosen_dim.append(temp_dim)
                    non_chosen_dim.remove(temp_dim)
                    self.__label[temp_dim] = False
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) < self.__region[temp_dim][0] or \
                            neg_set[remain_neg[r_i]].get_feature(temp_dim) > self.__region[temp_dim][1]:
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1
            # categorical
            else:
                chosen_dim.append(temp_dim)
                non_chosen_dim.remove(temp_dim)
                self.__label[temp_dim] = False
                r_i = 0
                while r_i < len(remain_neg):
                    if neg_set[remain_neg[r_i]].get_feature(temp_dim) != exa.get_feature(temp_dim):
                        remain_neg.remove(remain_neg[r_i])
                    else:
                        r_i += 1

        while len(non_chosen_dim) > self.__uncertain_bit:
            temp_dim = non_chosen_dim[self.__ro.get_uniform_integer(0, len(non_chosen_dim) - 1)]
            chosen_dim.append(temp_dim)
            non_chosen_dim.remove(temp_dim)
            self.__label[temp_dim] = False

        return dist_count

    # sequential Racos for mixed optimization
    # the dimension type includes float, integer and categorical
    def mix_opt(self, obj_fct=None, ss=2, bud=20, pn=1, rp=0.95, ub=1):

        sample_count = 0
        all_dist_count = 0

        # initialize sample set
        self.clear()
        self.log_clear()
        self.set_parameters(ss=ss, bud=bud, pn=pn, rp=rp, ub=ub)
        self.reset_model()
        self.initialize(obj_fct)

        # ------------------------------------------------------
        # print 'after initialization------------'
        # self.show_pos_pop()
        # self.show_pop()
        # ------------------------------------------------------

        # optimization
        budget_c = self.__sample_size + self.__positive_num
        while budget_c < self.__budget:
            budget_c += 1
            if budget_c % 10 == 0:
                # print '======================================================'
                print('budget ', budget_c, ':', self.__optimal.get_fitness())
                # self.__optimal.show_instance()
            while True:
                self.reset_model()
                chosen_pos = self.__ro.get_uniform_integer(0, self.__positive_num - 1)
                model_sample = self.__ro.get_uniform_double(0.0, 1.0)
                if model_sample <= self.__rand_probability:
                    dc = self.shrink_model(self.__pos_pop[chosen_pos], self.__pop)
                    all_dist_count += dc

                # -----------------------------------------------------------
                # self.show_region()
                # self.show_label()
                # -----------------------------------------------------------

                ins = self.pos_random_mix_isntance(self.__pos_pop[chosen_pos], self.__region, self.__label)

                sample_count += 1

                if (self.instance_in_list(ins, self.__pos_pop, self.__positive_num) is False) and (
                        self.instance_in_list(ins, self.__pop, self.__sample_size) is False):
                    ins.set_fitness(obj_fct(ins.get_features()))

                    # logging
                    model_instance, neg_set, new_instance = self.generate_environment(self.__pos_pop[chosen_pos], ins)
                    self.__model_ins.append(model_instance)
                    self.__negative_set.append(neg_set)
                    self.__new_ins.append(new_instance)

                    if ins.get_fitness() < self.__optimal.get_fitness():
                        self.__sample_label.append(1)
                    else:
                        self.__sample_label.append(0)
                    # ------------------------------------------------------
                    # print 'new sample:-------------------'
                    # ins.show_instance()
                    # print '------------------------------'
                    # ------------------------------------------------------

                    break
            self.online_update(ins)

            # ------------------------------------------------------
            # print 'after updating------------'
            # self.show_pos_pop()
            # self.show_pop()
            # ------------------------------------------------------

            self.update_optimal()
        print('average sample times of each sample:', float(sample_count) / self.__budget)
        print('average shrink times of each sample:', float(all_dist_count) / sample_count)
        return

    def _get_remaining_features(self):
        self.__remaining_features = []
        self.__generated_features = [x.get_features() for x in self.__generated]

        def is_same(c1, c2):
            if len(c1) != len(c2):
                return False
            else:
                n = len(c1)
                for i in range(n):
                    if c1[i] != c2[i]:
                        return False
            return True
        
        def cbnt_in_generated(cbnt):
            for one_generated in self.__generated_features:
                if is_same(cbnt, one_generated):
                    return True

            return False


        def dfs(cbnt, depth):
            if depth == self.__dimension.get_size():
                self.__all_cbnt_of_finite_space.append(cbnt)
            else:
                for num in range(self.__dimension.get_region(depth)[0], self.__dimension.get_region(depth)[1] + 1):
                    new_cbnt = cbnt.copy()
                    new_cbnt.append(num)
                    dfs(new_cbnt, depth+1)

        self.__all_cbnt_of_finite_space = []
        dfs([], 0)

        for cbnt in self.__all_cbnt_of_finite_space:
            if not cbnt_in_generated(cbnt):
                self.__remaining_features.append(cbnt)

        return self.__remaining_features

    
    def sample(self):
        if self.__reach_finite_threshhold:
            if len(self.__remaining_features) > 0:
                return self.__remaining_features.pop()
            else:
                raise SpaceRunOut()

        # initialization step
        if self.__sample_count < self.__sample_size + self.__positive_num:
            while True:
                if isinstance(self.__dim_capacity, int) and self.__generated_sample_count > self.__dim_capacity * 0.8:
                    self.__reach_finite_threshhold = True
                    self._get_remaining_features()

                    print(self.__generated_features)
                    print('--1--')
                    print(self.__remaining_features)

                    return self.__remaining_features.pop()

                self.reset_model()
                ins = self.random_instance(self.__dimension, self.__region, self.__label)
                if self.instance_in_list(ins, self.__generated, len(self.__generated)) is False:
                    self.__generated.append(ins)
                    break
        # optimization step
        else:
            while True:
                if isinstance(self.__dim_capacity, int) and self.__generated_sample_count > self.__dim_capacity * 0.8:
                    self.__reach_finite_threshhold = True
                    self._get_remaining_features()

                    print(self.__generated_features)
                    print('--2--')
                    print(self.__remaining_features)

                    return self.__remaining_features.pop()

                self.reset_model()
                chosen_pos = self.__ro.get_uniform_integer(0, self.__positive_num - 1)
                model_sample = self.__ro.get_uniform_double(0.0, 1.0)
                if model_sample <= self.__rand_probability:
                    dc = self.shrink_model(self.__pos_pop[chosen_pos], self.__pop)
                ins = self.pos_random_mix_isntance(self.__pos_pop[chosen_pos], self.__region, self.__label)
                if self.instance_in_list(ins, self.__generated, len(self.__generated)) is False:
                    self.__generated.append(ins)
                    break

        self.__generated_sample_count += 1
        
        return ins.get_features()

    # x is a list, f_x is the evaluation value of x
    def update_model(self, x, f_x):

        ins = Instance(self.__dimension)
        ins.set_features(x)
        ins.set_fitness(f_x)

        # initialization update
        if self.__sample_count < self.__sample_size + self.__positive_num:
            self.__pop.append(ins)
        # optimization update
        else:
            self.online_update(ins)
            self.update_optimal()

        self.__sample_count += 1

        # initialization process
        if self.__sample_count == self.__sample_size + self.__positive_num:
            self.__pop.sort(key=lambda instance: instance.get_fitness())
            for i in range(self.__positive_num):
                self.__pos_pop.append(self.__pop[0])
                del self.__pop[0]
            self.__optimal = self.__pos_pop[0].copy_instance()

        return
