import torch
import torch.nn as nn
import math


class NSMMPP(nn.Module):
    def __init__(self, label_size: int, hidden_size: int, target: int):
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
        self.W = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size)))
        self.U = nn.Parameter(self.init_weight(torch.empty(self.num_eq, hidden_size, hidden_size)))
        self.d = nn.Parameter(torch.zeros(self.num_eq, hidden_size))
        self.w = nn.Parameter(self.init_weight(torch.empty(label_size, hidden_size)))
        self.log_s = nn.Parameter(torch.zeros(label_size))
        self.debug = False

    def init_weight(self, w):
        stdv = 1. / math.sqrt(w.size()[-1])
        w.uniform_(-stdv, stdv)
        return w

    def scaled_softplus(self, x):
        s = torch.exp(self.log_s)
        return s * self.softplus(x / s)

    # all zeros
    def init_hidden(self):
        c_t = torch.zeros(self.hidden_size)
        c_ = torch.zeros_like(c_t)
        h_t = torch.zeros_like(c_t)
        hidden = (c_t, h_t, c_, None, None, None)
        return hidden

    # only compute hidden variables
    def forward_one_step(self, label_prev, label, t_prev, t, hidden):
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
        lambd_tilda = h.matmul(self.w.t())
        lambd = self.scaled_softplus(lambd_tilda)
        return lambd + 1e-9

    # compute NLL loss given a label_seq and a time_seq
    # sim_time_seq is simlulated times for computing integral
    def loglik(self, label_seq, time_seq, sim_time_seq, sim_time_idx):
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

    def forward(self, label_seq, time_seq, sim_time_seq, sim_time_idx):
        result = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx)
        return result[0].sum()

    def detect_outlier(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test, n_sample):
        with torch.no_grad():
            _, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx)
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

    def detect_outlier_instant(self, label_seq, time_seq, sim_time_seq, sim_time_idx, sim_time_diffs, time_test):
        with torch.no_grad():
            _, all_c, all_c_, all_delta, all_o, all_h_t, h_sim = self.loglik(label_seq, time_seq, sim_time_seq, sim_time_idx)
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
