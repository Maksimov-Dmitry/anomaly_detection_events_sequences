import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy
import hydra
from omegaconf import DictConfig
from src.entities.dataset_params import read_dataset_params, DatasetParams
import os
import pickle
from collections import defaultdict


def split_train_val(data, train_size):
    """
    Splits the given data into training and validation sets based on the 'id' key.

    Args:
        data (list of dict): The data to be split. Each dict should contain an 'id' key.
        train_size (float): Proportion of data to use for training. Should be between 0 and 1.

    Returns:
        tuple: Two lists containing training and validation data, respectively.
    """
    grouped_data = defaultdict(list)
    for d in data:
        grouped_data[d['id']].append(d)

    train_data = []
    val_data = []

    for _, id_group in grouped_data.items():
        n = len(id_group)
        n_train = int(n * train_size)

        train_data.extend(id_group[:n_train])  # Training set
        val_data.extend(id_group[n_train:])    # Validation set

    return train_data, val_data


def merge(a, b, idx=None):
    """
    Merges two sorted lists into a single sorted list and optionally keeps track of the origin of each element.

    Args:
        a (list): First sorted list.
        b (list): Second sorted list.
        idx (list, optional): List to store the index representing the origin of each element.

    Returns:
        list: A sorted list containing all elements from both input lists.
    """
    i = 0
    j = 0
    n = len(a)
    m = len(b)
    c = []
    while i < n and j < m:
        if a[i] <= b[j]:
            c.append(a[i])
            if idx is not None:
                idx.append(0)
            i += 1
        else:
            c.append(b[j])
            if idx is not None:
                idx.append(1)
            j += 1
    if i < n:
        c.extend(a[i:])
        if idx is not None:
            idx.extend([0] * len(a[i:]))
    elif j < m:
        c.extend(b[j:])
        if idx is not None:
            idx.extend([1] * len(b[j:]))
    return c


class MJPSim:
    """
    Base class for simulating Markov Jump Processes (MJP).

    Attributes:
        q (numpy.ndarray): Transition rate matrix.
        param (any): Additional parameters specific to the type of MJP.
    """
    def __init__(self, q, param):
        self.q = q
        self.param = param

    def sim(self, t_max, dt):
        vt_z, vz = self.sim_mjp(t_max=t_max)
        vt_event, lambda_x, t_x = self.sim_target(self.param, vt_z, vz, t_max=t_max, dt=dt)
        return {
            'start': 0,
            'stop': t_max,
            'time_context': vt_z,
            'mark_context': vz,
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'lambda_x': lambda_x,
            't_x': t_x,
        }

    def sim_mjp(self, t_max):
        q = self.q
        m = q.shape[0]
        assert (np.all(np.sum(q, axis=1) == 0))
        states = np.arange(0, m)
        stay = np.diag(q)
        trans = q.copy()
        np.fill_diagonal(trans, 0)
        trans = trans / np.sum(trans, axis=1)
        vt_z = [0]
        vz = [0]
        s = 0
        t = 0
        while True:
            t += np.random.exponential(-1 / stay[s])
            if t <= t_max:
                s = np.random.choice(states, p=trans[s, :])
                vt_z.append(t)
                vz.append(s)
            else:
                break
        return np.array(vt_z), np.array(vz, dtype=np.int32)

    def sim_target(self, param, vt_z, vz, t_max, dt):
        raise NotImplementedError

    def sim_next(self, lambda_t, lambda_max, t_beg, t_end):
        t_next = t_beg
        while True:
            t_next += np.random.exponential(1 / lambda_max)
            if (t_next > t_end) or (np.random.uniform() * lambda_max <= lambda_t(t_next)):
                return t_next


class PoisMJPSim(MJPSim):
    """
    Class for simulating Poisson MJP.
    Inherits from MJPSim.
    """
    def sim_target(self, lambda_, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        vt_event = []
        vt_z = np.append(vt_z, t_max)
        for k in range(len(vt_z) - 1):
            t_l = vt_z[k + 1]
            lambda_x[(t_x > t) & (t_x <= t_l)] = lambda_[vz[k]]
            lambda_t = lambda t: lambda_[vz[k]]
            lambda_max = lambda_[vz[k]]
            while True:
                t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                else:
                    break
            t = t_l
        vt_event = np.array(vt_event)
        return vt_event, lambda_x, t_x


class PoisSinMJPSim(MJPSim):
    def sim_target(self, param, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        vt_event = []
        vt_z = np.append(vt_z, t_max)
        for k in range(len(vt_z) - 1):
            t_l = vt_z[k + 1]
            idx = (t_x > t) & (t_x <= t_l)
            lambda_t = lambda t: param[vz[k]] * (1 + np.sin(t))
            lambda_x[idx] = lambda_t(t_x[idx])
            lambda_max = param[vz[k]]
            while True:
                t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                else:
                    break
            t = t_l
        vt_event = np.array(vt_event)
        return vt_event, lambda_x, t_x


class GamMJPSim(MJPSim):
    def sim_target(self, param, vt_z, vz, t_max, dt):
        t_x = np.arange(dt, t_max, dt)
        lambda_x = np.zeros_like(t_x)
        t = 0
        t_prev = 0
        vt_event = []
        vt_z = np.append(vt_z, t_max)
        lambda_event = []
        for k in range(len(vt_z) - 1):
            a = param[vz[k], 0]
            b = 1 / param[vz[k], 1]
            step = a * b
            if vt_z[k + 1] - t - step < 10 * dt:
                t_l = vt_z[k + 1]
            else:
                t_l = t + step
            while True:
                idx = (t_x > t) & (t_x <= t_l)

                def lambda_t(t):
                    return stats.gamma.pdf(t - t_prev, a, scale=b) / stats.gamma.sf(t - t_prev, a, scale=b)

                lambda_x[idx] = lambda_t(t_x[idx])
                if a >= 1:
                    lambda_max = lambda_t(t_l)
                else:
                    lambda_max = lambda_t(t)
                assert (lambda_max < np.inf)
                if lambda_max == 0:  # avoid overflow in exponential
                    t = t_l + 1
                else:
                    t = self.sim_next(lambda_t, lambda_max, t, t_l)
                if t <= t_l:
                    vt_event.append(t)
                    lambda_event.append(lambda_t(t))
                    t_prev = t
                elif t_l >= vt_z[k + 1]:
                    break
                else:
                    t = t_l
                    if vt_z[k + 1] - t - step < 10 * dt:
                        t_l = vt_z[k + 1]
                    else:
                        t_l = t + step
            t = t_l
        vt_event = np.array(vt_event)
        lambda_event = np.array(lambda_event)
        t_x = np.concatenate((t_x, vt_event))
        lambda_x = np.concatenate((lambda_x, lambda_event))
        idx = np.argsort(t_x)
        t_x = t_x[idx]
        lambda_x = lambda_x[idx]
        return vt_event, lambda_x, t_x


class OmissSim:
    """
    Class for simulating data with missing events (Omission).

    Attributes:
        w (float): Window size for generating test points.
        rate_omiss (float): Omission rate.
        regulator (function, optional): Function to modify the omission rate over time.
    """
    def __init__(self, w, rate_omiss=0.1, regulator=None):
        # regulator is a function which changes the rate over time
        self.rate_omiss = rate_omiss
        self.w = w
        self.regulator = regulator

    def sim(self, seq):
        vt_event = seq['time_target']
        t_max = seq['stop']
        t_min = seq['start']
        vt_event, vt_omiss = self.sim_omiss(vt_event, t_min)
        vt_test = self.gen_test(vt_event, t_min, t_max)
        vlabel = self.gen_label(vt_test, vt_omiss)
        seq = seq.copy()
        seq.update({
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'time_test': vt_test,
            'label_test': vlabel,
            'time_omiss': vt_omiss,
            'mark_omiss': np.ones_like(vt_omiss, dtype=np.int32),
        })
        return seq

    def sim_omiss(self, vt_event, t_min):
        n = len(vt_event)
        if self.regulator is None:
            rate = self.rate_omiss
        else:
            rate = self.rate_omiss * self.regulator(vt_event)
        trials = np.random.binomial(1, rate, n)
        # always keep the event at t_min
        if len(vt_event) > 0 and vt_event[0] == t_min:
            trials[0] = 0
        idx_omiss = np.nonzero(trials)
        vt_omiss = vt_event[idx_omiss]
        vt_event_left = np.delete(vt_event, idx_omiss)
        return vt_event_left, vt_omiss

    def gen_test(self, vt_event, t_min, t_max):
        w = self.w
        vt_test = []
        vt = vt_event
        # we ignore events at t_min but keep events at t_max
        if len(vt) > 0 and vt[0] == t_min:
            vt = np.concatenate((vt, [t_max]))
        else:
            vt = np.concatenate(([t_min], vt, [t_max]))
        n = len(vt)
        for i in range(n - 1):
            t = vt[i]
            vt_test.append(vt[i])
            while vt[i + 1] > t + w:
                t_next = t + np.random.uniform(0, w)
                vt_test.append(t_next)
                t = t_next
        vt_test = np.array(vt_test)
        return vt_test

    def gen_label(self, vt, vt_omiss):
        n = len(vt)
        vlabel = np.zeros(n - 1)
        for i in range(n - 1):
            t_beg = vt[i]
            t_end = vt[i + 1]
            if i == 0:
                vlabel[i] = np.any((vt_omiss >= t_beg) & (vt_omiss <= t_end))
            else:
                vlabel[i] = np.any((vt_omiss > t_beg) & (vt_omiss <= t_end))
        return vlabel


class CommissSim:
    """
    Class for simulating data with commission errors.

    Attributes:
        rate (float): Rate of commission errors.
        shrink (float): Factor to shrink the inter-event time. Defaults to 1.
        regulator (function, optional): Function to modify the commission rate over time.
    """
    def __init__(self, rate=0.1, shrink=1, regulator=None):
        self.rate = rate
        self.shrink = shrink
        self.regulator = regulator

    def sim(self, seq):
        vt_event = seq['time_target']
        t_max = seq['stop']
        t_min = seq['start']
        vt_event, vlabel = self.sim_commiss(vt_event, t_min, t_max)
        # skip the event at t_min
        vt_test = vt_event
        if vt_test[0] == t_min and vlabel[0] == 0:
            vt_test = vt_test[1:]
            vlabel = vlabel[1:]
        # padding
        vt_test = np.concatenate(([t_min], vt_test))
        seq = seq.copy()
        seq.update({
            'time_target': vt_event,
            'mark_target': np.ones_like(vt_event, dtype=np.int32),
            'time_test': vt_test,
            'label_test': vlabel,
        })
        return seq

    def sim_commiss(self, vt_event, t_min, t_max):
        rate = self.rate
        shrink = self.shrink
        if shrink < 1:
            inter_event = np.diff(np.concatenate((vt_event, [t_max])))
            inter_event *= shrink
            total_inter_event = inter_event.sum()
            m = np.random.poisson(total_inter_event * rate, 1)
            vt_commiss = np.random.uniform(0, total_inter_event, m)
            cum_inter_event = np.cumsum(inter_event)
            for i in range(m):
                j = np.argwhere(cum_inter_event > vt_commiss[i])[0]
                if j > 0:
                    tmp = vt_commiss[i] - cum_inter_event[j - 1]
                else:
                    tmp = vt_commiss[i]
                tmp += vt_event[j]
                assert (tmp >= vt_event[0])
                vt_commiss[i] = tmp
        else:
            m = np.random.poisson((t_max - t_min) * rate, 1)
            vt_commiss = np.random.uniform(t_min, t_max, m)
            if self.regulator is not None:
                p = self.regulator(vt_commiss)
                keep = (np.random.binomial(1, p) > 0)
                vt_commiss = vt_commiss[keep]
        vt_commiss = np.sort(vt_commiss)
        vlabel = []
        vt_event = np.array(merge(vt_event, vt_commiss, vlabel))
        vlabel = np.array(vlabel)
        return vt_event, vlabel


def compute_empirical_rate(seqs):
    """
    Computes the empirical rate of events for a list of sequences.

    Args:
        seqs (list of dict): List of sequences. Each sequence is a dictionary containing 'start' and 'stop' keys.

    Returns:
        float: The computed empirical rate.
    """
    t = 0
    n = 0
    for seq in seqs:
        t += seq['stop'] - seq['start']
        n += len(seq['time_target'])
    return n / t


def sim_data_test_omiss(data_train, data_test, p=0.1, seed=0, regulator=None, regulator_generator=None):
    """
    Simulates test data with omission errors.

    Args:
        data_train (list of dict): Training data.
        data_test (list of dict): Test data.
        p (float): Omission probability.
        seed (int): Random seed.
        regulator (function, optional): Function to modify the omission rate.
        regulator_generator (function, optional): Function to generate a new regulator function for each test.

    Returns:
        list of dict: Test data with omission errors.
    """
    np.random.seed(seed)
    data_test_omiss = copy.deepcopy(data_test)
    n_test = len(data_test)
    w = 2 / compute_empirical_rate(data_train)
    if regulator_generator is None:
        omiss_sim = OmissSim(w, p, regulator=regulator)
        for i in range(n_test):
            data_test_omiss[i] = omiss_sim.sim(data_test_omiss[i])
    else:
        for i in range(n_test):
            regulator = regulator_generator()
            omiss_sim = OmissSim(w, p, regulator=regulator)
            data_test_omiss[i] = omiss_sim.sim(data_test_omiss[i])
    return data_test_omiss


def sim_data_test_commiss(data_test, alpha=0.1, seed=0, regulator=None, regulator_generator=None):
    """
    Simulates test data with commission errors.

    Args:
        data_test (list of dict): Test data.
        alpha (float): Commission rate factor.
        seed (int): Random seed.
        regulator (function, optional): Function to modify the commission rate.
        regulator_generator (function, optional): Function to generate a new regulator function for each test.

    Returns:
        list of dict: Test data with commission errors.
    """
    np.random.seed(seed)
    data_test_commiss = copy.deepcopy(data_test)
    n_test = len(data_test)
    rate = compute_empirical_rate(data_test)
    if regulator_generator is None:
        commiss_sim = CommissSim(alpha * rate, 1, regulator=regulator)
        for i in range(n_test):
            data_test_commiss[i] = commiss_sim.sim(data_test_commiss[i])
    else:
        for i in range(n_test):
            regulator = regulator_generator()
            commiss_sim = CommissSim(alpha * rate, 1, regulator=regulator)
            data_test_commiss[i] = commiss_sim.sim(data_test_commiss[i])
    return data_test_commiss


def create_rand_pc_regulator(step, t_min, t_max):
    """
    Creates a random piecewise constant regulator function.

    Args:
        step (float): Step size for each piece.
        t_min (float): Minimum time value.
        t_max (float): Maximum time value.

    Returns:
        function: A piecewise constant regulator function.
    """
    m = np.floor((t_max - t_min) / step).astype(int)
    p = np.random.uniform(size=m)

    def regulator(t):
        i = np.floor((t - t_min) / step).astype(int)
        return p[i]

    return regulator


def plot_events(seq):
    """
    Plots events and their intensity function.

    Args:
        seq (dict): A dictionary containing keys like 'time_target', 'lambda_x', 't_x', etc.

    Returns:
        None: This function only produces a plot.
    """
    vt_event = seq['time_target']
    lambda_x = seq['lambda_x']
    t_x = seq['t_x']
    vt_omiss = seq.get('time_omiss')
    scale = 0.25 * np.max(lambda_x)
    plt.figure()
    if vt_omiss is None:
        vlabel = seq.get('label_test')
        if vlabel is None:
            plt.plot(t_x, lambda_x)
            plt.stem(vt_event, scale * np.ones_like(vt_event), 'k-', 'ko')
        else:
            plt.plot(t_x, lambda_x)
            plt.stem(vt_event[vlabel == 0], scale * np.ones_like(vt_event[vlabel == 0]), 'k-', 'ko')
            if any(vlabel):
                plt.stem(vt_event[vlabel == 1], scale * np.ones_like(vt_event[vlabel == 1]), 'r-', 'ro')
    else:
        plt.plot(t_x, lambda_x)
        if len(vt_event) > 0:
            plt.stem(vt_event, scale * np.ones_like(vt_event), 'k-', 'ko')
        if len(vt_omiss) > 0:
            plt.stem(vt_omiss, scale * np.ones_like(vt_omiss), 'r-', 'ro')


def create_dataset(dataset_params: DatasetParams):
    """
    Creates and saves synthetic datasets based on the given parameters.

    Args:
        dataset_params (DatasetParams): An object containing all the necessary parameters to create the dataset.
    """
    folder = f'data/raw/{dataset_params.process}'
    os.makedirs(folder, exist_ok=True)
    Q = np.array([
        [-0.05, 0.05],
        [0.05, -0.05]
    ])
    data_train = []
    data_test = []
    params = []
    if dataset_params.process == 'pois':
        for i in range(dataset_params.n_persons):
            np.random.seed(dataset_params.seed * dataset_params.n_persons + i)
            param = np.random.uniform(dataset_params.context_intensity_pois_min,
                                      np.array(dataset_params.context_intensity_pois_min) * np.array(dataset_params.context_intensity_pois_factors),
                                      len(dataset_params.context_intensity_pois_min))
            params.append(param)
            sim = PoisMJPSim(q=Q, param=param)
            for _ in range(dataset_params.n_train):
                data = sim.sim(dataset_params.t_max, dataset_params.dt)
                data['id'] = i
                data_train.append(data)
            for _ in range(dataset_params.n_test):
                data = sim.sim(dataset_params.t_max, dataset_params.dt)
                data['id'] = i
                data_test.append(data)
    with open(f'{folder}/param.pkl', 'wb') as f:
        pickle.dump(params, f)
    with open(f'{folder}/train.pkl', 'wb') as f:
        pickle.dump(data_train, f)
    with open(f'{folder}/test.pkl', 'wb') as f:
        pickle.dump(data_test, f)

    data_test_omiss = sim_data_test_omiss(data_train, data_test, dataset_params.non_regulator_outliers_prob, dataset_params.seed)
    with open(f'{folder}/test_omiss_{dataset_params.non_regulator_outliers_prob}.pkl', 'wb') as f:
        pickle.dump(data_test_omiss, f)
    data_test_commiss = sim_data_test_commiss(data_test, dataset_params.non_regulator_outliers_prob, dataset_params.seed)
    with open(f'{folder}/test_commiss_{dataset_params.non_regulator_outliers_prob}.pkl', 'wb') as f:
        pickle.dump(data_test_commiss, f)


@hydra.main(version_base=None, config_path="../../configs", config_name="dataset_config")
def create_dataset_command(cfg: DictConfig):
    params = read_dataset_params(cfg)
    create_dataset(params)


if __name__ == '__main__':
    create_dataset_command()
