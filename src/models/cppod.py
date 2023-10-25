import torch
import torch.nn as nn
import math


class NSMMPP(nn.Module):
    """
    Neural Stochastic Modeling of Mixed Point Process (NSMMPP).

    This model utilizes, embedding layers for events or contexts and personalisation and a series of equations for temporal point processes.
    It leverages different activations in the transformation and update of hidden states. Intensity functions are derived from hidden states.

    Attributes:
        target (int): Specific event type or mark of interest.
        sigmoid (nn.Module): Sigmoid activation function for neural transformations.
        tanh (nn.Module): Hyperbolic tangent activation function for neural transformations.
        softplus (nn.Module): Softplus activation function to achieve a smooth approximation of ReLU.
        label_size (int): Total number of unique event labels or contexts.
        hidden_size (int): Dimensionality of hidden states in the model.
        num_eq (int): Total number of equations used in the model.
        Emb (nn.Parameter): Tensor representing the general embeddings for different event labels.
        EmbPersonalised (nn.Parameter): Embedding tensor personalized for specific entities or users.
        W, U, d, w, log_s (nn.Parameter): Parameters corresponding to the weights, biases, and scaling factors in the equations.

    Methods:
        forward_one_step: Compute the hidden states for a single time step given previous states and event information.
        h_to_lambd: Maps the computed hidden states to corresponding intensity values.
        loglik: Calculates the negative log-likelihood of given sequences of events and times.
        forward: Main forward pass for a given sequence of events.
    """

    def __init__(self, label_size: int, hidden_size: int, target: int, number_of_persons):
        super(NSMMPP, self).__init__()
        self.target = target
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.num_eq = 7
        # add a special event label for initialization
        self.Emb = nn.Parameter(self.init_weight(torch.empty(hidden_size, label_size + 1)))
        if number_of_persons is not None:
            self.EmbPersonalised = nn.Parameter(self.init_weight(torch.empty(hidden_size, number_of_persons)))
        self.W = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size)))
        self.U = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size)))
        self.d = nn.Parameter(torch.zeros(self.num_eq, hidden_size))
        self.w = nn.Parameter(self.init_weight(torch.empty(label_size, hidden_size)))
        self.log_s = nn.Parameter(torch.zeros(label_size))

    def init_weight(self, w):
        stdv = 1. / math.sqrt(w.size()[-1])
        w.uniform_(-stdv, stdv)
        return w

    def scaled_softplus(self, x):
        s = torch.exp(self.log_s)
        return s * self.softplus(x / s)

    def init_hidden(self):
        c_t = torch.zeros(self.hidden_size)
        c_ = torch.zeros_like(c_t)
        h_t = torch.zeros_like(c_t)
        hidden = (c_t, h_t, c_, None, None, None)
        return hidden

    def forward_one_step(self, label_prev, label, t_prev, t, hidden):
        """
        Forward computation for one time step. Only compute hidden variables

        Args:
            label_prev (tensor): Previous label.
            label (tensor): Current label.
            t_prev (float): Previous time.
            t (float): Current time.
            hidden (tuple): Previous hidden states.

        Returns:
            tuple: Updated hidden states.
        """
        c_t, h_t, c_, _, _, _ = hidden
        temp = self.W.matmul(label_prev) + self.U.matmul(h_t) + self.d
        i = self.sigmoid(temp[0, :])
        f = self.sigmoid(temp[1, :])
        z = self.tanh(temp[2, :])
        o = self.sigmoid(temp[3, :])
        i_ = self.sigmoid(temp[4, :])
        f_ = self.sigmoid(temp[5, :])
        delta = self.softplus(temp[6, :])
        c = f * c_t + i * z
        c_ = f_ * c_ + i_ * z
        c_t = c_ + (c - c_) * torch.exp(-delta * (t - t_prev))
        h_t = o * self.tanh(c_t)
        hidden = (c_t, h_t, c_, c, delta, o)
        return hidden

    def h_to_lambd(self, h):
        """
        Map hidden states to lambda (intensity function values).

        Args:
            h (tensor): Hidden state tensor.

        Returns:
            tensor: Lambda tensor.
        """
        lambd_tilda = h.matmul(self.w.t())
        lambd = self.scaled_softplus(lambd_tilda)
        return lambd + 1e-9

    def loglik(self, label_seq, time_seq, sim_time_seq, sim_time_idx, seq_id):
        """
        Compute negative log-likelihood. Sim_time_seq is simlulated times for computing integral.

        Args:
            label_seq (tensor): Label sequence.
            time_seq (tensor): Time sequence.
            sim_time_seq (tensor): Simulated time sequence.
            sim_time_idx (tensor): Simulated time index.
            seq_id (int): Sequence ID.

        Returns:
            tuple: Computed values including log-likelihood.
        """
        n = len(time_seq)
        # collect states right after each event
        # last event is EOS marker
        all_c = torch.zeros(n - 1, self.hidden_size)
        all_c_ = torch.zeros_like(all_c)
        all_delta = torch.zeros_like(all_c)
        all_o = torch.zeros_like(all_c)
        all_h_t = torch.zeros_like(all_c)
        hidden = self.init_hidden()
        # BOS event is 0 at time 0
        if hasattr(self, 'EmbPersonalised'):
            label_prev = self.EmbPersonalised[:, seq_id].squeeze()
        else:
            label_prev = self.Emb[:, label_seq[0]].squeeze()
        t_prev = time_seq[0]
        for i in range(1, n):
            label = self.Emb[:, label_seq[i]].squeeze()
            t = time_seq[i]
            hidden = self.forward_one_step(label_prev, label, t_prev, t, hidden)
            _, all_h_t[i - 1, :], all_c_[i - 1, :], all_c[i - 1, :], all_delta[i - 1, :], all_o[i - 1, :] = hidden
            label_prev = label
            t_prev = t
        beg = 0
        target = self.target
        h_t = all_h_t[beg:-1, :]
        if h_t.shape[0] > 0:
            lambd = self.h_to_lambd(h_t)
            term1 = (lambd[label_seq[(1 + beg):-1] == target, target - 1]).log().sum()
        else:
            term1 = 0
        c_sim = (all_c_[sim_time_idx, :] +
                 (all_c[sim_time_idx, :] - all_c_[sim_time_idx, :]) *
                 torch.exp(-all_delta[sim_time_idx, :] * (sim_time_seq - time_seq[sim_time_idx])[:, None])
                 )
        h_sim = all_o[sim_time_idx, :] * self.tanh(c_sim)
        lambd_sim = self.h_to_lambd(h_sim)
        term2 = lambd_sim[:, target - 1].mean() * (time_seq[-1] - time_seq[0])
        loglik = term1 - term2

        return -loglik, all_c, all_c_, all_delta, all_o, all_h_t, h_sim

    def forward(self, label_seq, time_seq, sim_time_seq, sim_time_idx, seq_id):
        result = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx, seq_id)
        return result[0].sum()

    def detect_outlier(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test, id, n_sample):
        """
        Detect outliers using the trained model.

        Args:
            label_seq (tensor): Label sequence.
            time_seq (tensor): Time sequence.
            sim_time_seq (tensor): Simulated time sequence.
            sim_time_idx (tensor): Simulated time index.
            sim_time_diffs (tensor): Simulated time differences.
            time_test (tensor): Test time sequence.
            id (int): Sequence ID.
            n_sample (int): Number of samples for Monte Carlo integration.

        Returns:
            tuple: Detection scores and other outputs.
        """
        with torch.no_grad():
            _, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx, id)
            n = len(time_test)
            m = len(time_seq)
            score = torch.zeros(n - 1)
            target = self.target
            j = 0
            ts = torch.zeros(n_sample)
            for i in range(n - 1):
                t_beg = time_test[i]
                t_end = time_test[i + 1]
                ts.uniform_(t_beg, t_end)
                # find the first event after t_beg
                while j < m and time_seq[j] <= t_beg:
                    j += 1
                assert (time_seq[j] > t_beg)
                assert (time_seq[j - 1] <= t_beg)
                Lambd = 0
                k = j
                # calculate Lambda piecewise segmented by events
                while k < m and time_seq[k - 1] <= t_end:
                    ts_in_range = ts[(ts > time_seq[k - 1]) & (ts <= time_seq[k])]
                    if len(ts_in_range) > 0:
                        c = all_c[k - 1, :]
                        c_ = all_c_[k - 1, :]
                        delta = all_delta[k - 1, :]
                        o = all_o[k - 1, :]
                        c_ts = c_ + (c - c_) * torch.exp(-delta[None, :] * (ts_in_range[:, None] - time_seq[k - 1]))
                        h_ts = o * self.tanh(c_ts)
                        lambd_all = self.h_to_lambd(h_ts)
                        lambd = lambd_all[:, target - 1]
                        Lambd += lambd.sum() / n_sample
                    k += 1
                Lambd *= (t_end - t_beg)
                score[i] = Lambd
            lambd = self.h_to_lambd(all_h_t)[:-1]
            lambd_sim = self.h_to_lambd(h_sim)
            return score, -score, lambd, lambd_sim

    def detect_outlier_instant(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test, id):
        """
        Detect outliers instantly.

        Args:
            label_seq (tensor): Label sequence.
            time_seq (tensor): Time sequence.
            sim_time_seq (tensor): Simulated time sequence.
            sim_time_idx (tensor): Simulated time index.
            sim_time_diffs (tensor): Simulated time differences.
            time_test (tensor): Test time sequence.
            id (int): Sequence ID.

        Returns:
            tuple: Detection scores and other outputs.
        """
        with torch.no_grad():
            _, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx, id)
            n = len(time_test)
            m = len(time_seq)
            score = torch.zeros(n - 1)
            target = self.target
            j = 0
            for i in range(n - 1):
                t_end = time_test[i + 1]
                if t_end == 0:
                    j = 1
                else:
                    # find the first event at/after t_end
                    while j < m and time_seq[j] < t_end:
                        j += 1
                    assert (time_seq[j] >= t_end)
                    # last event before t_end
                    assert (time_seq[j - 1] < t_end)
                c = all_c[j - 1, :]
                c_ = all_c_[j - 1, :]
                delta = all_delta[j - 1, :]
                o = all_o[j - 1, :]
                c_ts = c_ + (c - c_) * torch.exp(-delta * (t_end - time_seq[j - 1]))
                h_ts = o * self.tanh(c_ts)
                lambd_all = self.h_to_lambd(h_ts)
                lambd = lambd_all[target - 1]
                score[i] = lambd
            lambd = self.h_to_lambd(all_h_t)[:-1]
            lambd_sim = self.h_to_lambd(h_sim)
            return score, -score, lambd, lambd_sim
