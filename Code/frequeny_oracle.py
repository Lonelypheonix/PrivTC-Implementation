import numpy as np
import math
import xxhash


class OUE:  # OUE has the same variance as OLH
    def __init__(self, domain_size=None, epsilon=None, user_num=None, sampling_factor=1):
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.user_num = user_num

        self.group_user_num = 0
        self.sampling_factor = sampling_factor

        self.perturbed_count = np.zeros(self.domain_size, dtype=int)
        self.aggregated_count = np.zeros(self.domain_size, dtype=int)
        self.aggregated_prob = np.zeros(self.domain_size, dtype=float)
        self.p = 0.5
        self.q = 1.0 / (math.exp(self.epsilon) + 1)

    def operation_perturb(self, real_value=None):
        self.perturbed_count[real_value] += 1
        return

    def operation_aggregate(self):
        tmp_perturbed_count_1 = np.copy(self.perturbed_count)
        est_count = np.random.binomial(tmp_perturbed_count_1, self.p)
        tmp_perturbed_count_0 = self.group_user_num - np.copy(self.perturbed_count)
        est_count += np.random.binomial(tmp_perturbed_count_0, self.q)
        a = 1.0 / (self.p - self.q)
        b = self.group_user_num * self.q / (self.p - self.q)
        est_count = a * est_count - b
        self.aggregated_count = est_count / self.group_user_num * self.user_num
        self.aggregated_prob = est_count / self.group_user_num
        return


class OLH:
    def __init__(self, domain_size=None, epsilon=None, user_num=None, sampling_factor=1):
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.user_num = user_num
        self.group_user_num = 0
        self.perturbed_count = np.zeros(self.domain_size, dtype=int)
        self.aggregated_count = np.zeros(self.domain_size, dtype=int)
        self.sampling_factor = sampling_factor
        self.ee = np.exp(self.epsilon)
        self.g = int(round(self.ee)) + 1
        self.p = self.ee / (self.ee + self.g - 1)
        self.q = 1 / self.g
        self.var = 4 * self.ee / (self.ee - 1) ** 2
        self.user_real_val_list = []

    def operation_perturb(self, real_value=None):
        self.perturbed_count[real_value] += 1
        self.user_real_val_list.append(real_value)
        return

    def operation_aggregate(self):
        assert self.group_user_num == len(self.user_real_val_list)
        samples_one = np.random.random_sample(self.group_user_num)
        hash_function_seeds_list = np.random.randint(0, self.group_user_num, self.group_user_num)
        hashed_val_list = np.zeros(self.group_user_num, dtype=int)
        reported_val_list = np.zeros(self.group_user_num, dtype=int)
        est_count = np.zeros(self.domain_size, dtype=int)
        for i in range(self.group_user_num):
            tmp_real_val = self.user_real_val_list[i]
            hashed_val_list[i] = (xxhash.xxh32(str(tmp_real_val), seed=hash_function_seeds_list[i]).intdigest()) % self.g
            if samples_one[i] > self.p:
                tmp_report_val = np.random.randint(0, self.g - 1)
                if tmp_report_val >= hashed_val_list[i]:
                    tmp_report_val += 1
            else:
                tmp_report_val = hashed_val_list[i]
            reported_val_list[i] = tmp_report_val

        for j in range(self.domain_size):
            for i in range(self.group_user_num):
                hashed_j = (xxhash.xxh32(str(j), seed=hash_function_seeds_list[i]).intdigest()) % self.g
                if hashed_j == reported_val_list[i]:
                    est_count[j] += 1

        a = 1.0 * self.g / (self.p * self.g - 1)
        b = 1.0 * self.group_user_num / (self.p * self.g - 1)
        est_count = a * est_count - b
        self.aggregated_count = est_count / self.group_user_num * self.user_num
        self.aggregated_prob = est_count / self.group_user_num
        return
