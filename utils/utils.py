import os
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)
class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        self.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

from torch.nn.utils.rnn import pad_sequence
import re
import collections

#from torch._six import string_classes # torch._six does not exist in latest pytorch !!
string_classes = (str,)

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def variable_len_collate(batch, batch_first=True, padding_value=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        numel = [x.numel() for x in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            storage = elem.untyped_storage()._new_shared(sum(numel))
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out) if np.all(numel[0] == numel) else pad_sequence(batch,
                                                                                             batch_first=batch_first,
                                                                                             padding_value=padding_value)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return variable_len_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: variable_len_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(variable_len_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [variable_len_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

import pickle
import torch.distributed as dist


def distributed_sync_dict(array, world_size, rank, device, MAX_LENGTH=10*2**20): # default MAX_LENGTH = 10MB
    def _pack_data(_array):
        data = pickle.dumps(_array)
        data_length = int(len(data))
        data = data_length.to_bytes(4, "big") + data
        assert len(data) < MAX_LENGTH
        data += bytes(MAX_LENGTH - len(data))
        data = np.frombuffer(data, dtype=np.uint8)
        assert len(data) == MAX_LENGTH
        return torch.from_numpy(data)
    def _unpack_data(_array):
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        return pickle.loads(data[4:data_length+4])
    def _unpack_size(_array):
        print(_array.shape, _array[:4])
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        print(data_length,data[:4])
        return data_length

    # prepare output buffer
    output_tensors = [torch.zeros(MAX_LENGTH, dtype=torch.uint8, device=device) for _ in range(world_size)]
    # pack data using pickle into input/output
    output_tensors[rank][:] = _pack_data(array)

    # sync data
    dist.all_gather(output_tensors, output_tensors[rank])

    # unpack data and merge into single dict
    return {id:val for array_tensor in output_tensors for id,val in _unpack_data(array_tensor).items()}