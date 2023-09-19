from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from functools import partial


class CPPODDataset(Dataset):
    def __init__(self, dataset, target):
        self.target = target
        self.dataset = self.convert(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def convert(self, seqs):
        if seqs is None:
            return None
        m_t = self.target + 1

        def _convert(seq):
            seq_id = seq.get('id')
            time_c = seq['time_context']
            mark_c = seq['mark_context']
            start = seq['start']
            stop = seq['stop']
            time_t = seq['time_target']
            mark_t = seq['mark_target']
            assert np.all(time_c == np.sort(time_c))
            assert np.all(time_t == np.sort(time_t))
            time = []
            mark = []
            i_c = 0
            i_t = 0
            n_c = len(time_c)
            n_t = len(time_t)
            assert (n_c == len(mark_c))
            assert (n_t == len(mark_t))
            while i_c < n_c and i_t < n_t:
                if time_t[i_t] <= time_c[i_c]:
                    time.append(time_t[i_t])
                    mark.append(mark_t[i_t])
                    i_t += 1
                else:
                    time.append(time_c[i_c])
                    mark.append(m_t + mark_c[i_c])
                    i_c += 1
            if i_t < n_t:
                time.extend(time_t[i_t:])
                mark.extend(mark_t[i_t:])
            if i_c < n_c:
                time.extend(time_c[i_c:])
                mark.extend(m_t + mark_c[i_c:])
            return {
                'id': seq_id,
                'time': time,
                'mark': mark,
                'start': start,
                'stop': stop,
                'time_test': seq.get('time_test'),
                'label_test': seq.get('label_test'),
                'lambda_x': seq.get('lambda_x'),
                't_x': seq.get('t_x'),
            }
        return [_convert(seq) for seq in seqs]


def collate_fn(batch, multiple, diff_sample_size=100, regular=False, step=None):
    if batch is None:
        return None, None, None
    output = []
    for seq in batch:
        label_seq = torch.tensor(
            np.concatenate(([0], seq['mark'], [0])),
            dtype=torch.long,
        )
        time_seq = torch.tensor(
            np.concatenate(([seq['start']], seq['time'], [seq['stop']])),
            dtype=torch.float,
        )
        n = len(time_seq)
        t0 = seq['start']
        tn = seq['stop']
        if regular:
            sim_time_seq = torch.arange(t0, tn, step)
            sim_time_idx = torch.zeros_like(sim_time_seq, dtype=torch.long)
        else:
            sim_time_seq = time_seq.new_empty(n * multiple)
            sim_time_seq.uniform_(t0, tn)
            sim_time_idx = label_seq.new_zeros(n * multiple)
        for j in range(n - 1):
            sim_time_idx[(sim_time_seq > time_seq[j]) &
                         (sim_time_seq <= time_seq[j + 1])
                         ] = j
        temp = sim_time_seq.new_empty(diff_sample_size)
        temp.exponential_(1)
        sim_time_diffs, _ = torch.sort(temp)
        if seq['time_test'] is None:
            time_test = None
        else:
            time_test = torch.tensor(
                seq['time_test'],
                dtype=torch.float)
        if seq['label_test'] is None:
            label_test = None
        else:
            label_test = torch.tensor(
                seq['label_test'],
            )
        item = {
            'id': seq['id'],
            'label_seq': label_seq,
            'time_seq': time_seq,
            'sim_time_seq': sim_time_seq,
            'sim_time_idx': sim_time_idx,
            'sim_time_diffs': sim_time_diffs,
            'time_test': time_test,
            'label_test': label_test,
            'lambda_x': seq['lambda_x'],
            't_x': seq['t_x'],
        }
        output.append(item)
    return output


class NonContextCPPODDataset(CPPODDataset):
    def convert(self, seqs):
        if seqs is None:
            return None

        def _convert(seq):
            seq_id = seq.get('id')
            start = seq['start']
            stop = seq['stop']
            time_t = seq['time_target']
            mark_t = seq['mark_target']
            assert np.all(time_t == np.sort(time_t))
            assert len(time_t) == len(mark_t)
            return {
                'id': seq_id,
                'time': time_t,
                'mark': mark_t,
                'start': start,
                'stop': stop,
                'time_test': seq.get('time_test'),
                'label_test': seq.get('label_test'),
                'lambda_x': seq.get('lambda_x'),
                't_x': seq.get('t_x'),
            }
        return [_convert(seq) for seq in seqs]


def get_cppod_dataloader(dataset: list, target: int, sample_multiplier: int, use_context: bool):
    if use_context:
        set = CPPODDataset(dataset, target)
    else:
        set = NonContextCPPODDataset(dataset, target)
    collate_fn_fixed = partial(
        collate_fn,
        multiple=sample_multiplier,
    )
    batch_gen = DataLoader(set, batch_size=1, shuffle=False, collate_fn=collate_fn_fixed)
    return batch_gen
