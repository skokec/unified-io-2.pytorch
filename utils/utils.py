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


####################################################################################################################
# General class that can merge split image results - can merge tensor data (image or heatmap) and points
class ImageGridCombiner:
    class Data:
        def __init__(self, w,h):
            self.full_size = [h,w]
            self.full_data = dict()
            self.image_names = []
        def add_image_name(self, name):
            self.image_names.append(name)

        def get_image_name(self):
            org_names = ["_patch".join(n.split("_patch")[:-1]) + os.path.splitext(n)[1] for n in self.image_names]
            # check that all images had the same original name
            org_names = np.unique(org_names)

            if len(org_names) != 1:
                raise Exception("Invalid original names found: %s" % ",".join(org_names))

            return org_names[0]

        def set_tensor2d(self, name, partial_data, roi_x, roi_y, merge_op=None):
            if name not in self.full_data:
                # store data on CPU for large images to avoid excesive memory usage
                dev = partial_data.device if np.prod(self.full_size) < 2500*2500 else 'cpu'
                self.full_data[name] = torch.zeros(list(partial_data.shape[:-2]) + self.full_size, dtype=partial_data.dtype,
                                                   device=dev)

            full_data_roi = self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
            partial_data_roi = partial_data[..., 0:(roi_y[1] - roi_y[0]),0:(roi_x[1] - roi_x[0])].type(full_data_roi.dtype)
            try:
                # default merge operator is to use max unless specified otherwise
                if merge_op is None:
                    merge_op = lambda Y,X: torch.where(Y.abs() < X.abs(), X, Y)

                self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] = merge_op(full_data_roi, partial_data_roi.to(self.full_data[name].device))
            except:
                print('error')

        def set_instance_description(self, name, partial_instance_list, partial_instance_mask, reverse_desc, x, y):
            if name not in self.full_data:
                self.full_data[name] = ([],[])

            if len(partial_instance_list) > 0:
                if np.all([i != 0 for i,desc in partial_instance_list.items()]):
                    all_ids = partial_instance_mask.nonzero().cpu().numpy()
                    all_values = partial_instance_mask[all_ids[:,0],all_ids[:,1]].cpu().numpy()
                else:
                    all_ids, all_values = None, None
                for i,desc in partial_instance_list.items():
                    if all_ids is None or all_values is None:
                        # revert to slower but more correct version
                        i_mask = (partial_instance_mask == i).nonzero().cpu().numpy()
                    else:
                        i_mask = all_ids[all_values == i,:] if len(all_ids) > 0 else np.zeros((0,2))

                    self.full_data[name][0].append(desc + np.array(([y,x] if reverse_desc else [x,y]) + [0]*(len(desc)-2)))
                    self.full_data[name][1].append(i_mask + np.array([y, x]))

        def get(self, data_names):
            return [self.full_data.get(n) for n in data_names]

        def get_instance_description(self, name, out_shape, out_mask_tensor, overlap_thr=0, merge_dist_thr=0):
            instance_list, instance_mask_ids = self.full_data[name]

            # clip mask ids to out_mask_tensor.shape
            instance_mask_ids = [np.array([c for c in i_mask if c[0] >= 0 and c[1] >= 0 and c[0] < out_shape[0] and c[1] < out_shape[1]])
                                        for i_mask in instance_mask_ids]

            instance_indexes = [set(np.ravel_multi_index((np.array(i_mask)[:, 0], np.array(i_mask)[:, 1]), dims=out_shape)) if len(i_mask) > 0 else set()
                                        for i_mask in instance_mask_ids]

            retained = self._find_retained_instances(instance_indexes, overlap_thr)

            if merge_dist_thr > 0:
                for i, x_i in enumerate(instance_list):
                    if retained[i]:
                        for j, x_j in enumerate(instance_list):
                            if j > i and retained[j]:
                                dist = np.sqrt(np.sum(np.abs(x_i[:2]-x_j[:2])**2))
                                if dist < merge_dist_thr:
                                    retained[j] = False

            instance_list = [x for i, x in enumerate(instance_list) if retained[i]]
            instance_mask_ids = [x for i, x in enumerate(instance_mask_ids) if retained[i]]
            instance_indexes = [x for i, x in enumerate(instance_indexes) if retained[i]]

            instance_dict = {}
            instance_indexes_dict = {}

            for id, (i_center, i_mask, i_mask_indexes) in enumerate(zip(instance_list, instance_mask_ids, instance_indexes)):
                i_mask_ids = torch.from_numpy(i_mask)
                if len(i_mask_ids) > 0:
                    if out_mask_tensor is not None:
                        out_mask_tensor[(i_mask_ids[:, 0], i_mask_ids[:, 1])] = id + 1
                instance_dict[id + 1] = i_center
                instance_indexes_dict[id+1] = list(i_mask_indexes)

            return instance_dict, out_mask_tensor, instance_indexes_dict

        def _find_retained_instances(self, instance_indexes, overlap_thr):
            retained = np.ones(shape=len(instance_indexes), dtype=bool)
            for i in range(len(instance_indexes)):
                if retained[i]:
                    for j in range(i + 1, len(instance_indexes)):
                        inter_ij = len(instance_indexes[i].intersection(instance_indexes[j]))
                        iou_ratio = inter_ij / (len(instance_indexes[i]) + len(instance_indexes[j]) - inter_ij + 1e-5)
                        if iou_ratio > overlap_thr:
                            if len(instance_indexes[i]) > len(instance_indexes[j]):
                                retained[j] = False
                            else:
                                retained[i] = False
                                break
            return retained
    def __init__(self):
        self.current_index = None
        self.current_data = None

    def add_image(self, im_name, grid_index, data_map_tensor, data_map_instance_desc, custom_merge_ops={}):
        n, x, y, w, h, org_w, org_h = grid_index

        # set roi and clamp to max size
        roi_x = x, min(x + w, org_w)
        roi_y = y, min(y + h, org_h)

        finished_data = None

        if self.current_index is None or n != self.current_index:
            finished_data = self.current_data
            self.current_data = self.Data(org_w, org_h)
            self.current_index = n

        self.current_data.add_image_name(im_name)

        if n == self.current_index:
            for name,partial_data in data_map_tensor.items():
                if partial_data is not None:
                    self.current_data.set_tensor2d(name, partial_data, roi_x, roi_y, merge_op=custom_merge_ops.get(name))

            for name,partial_data in data_map_instance_desc.items():
                if partial_data is not None:
                    self.current_data.set_instance_description(name, partial_data[0], partial_data[1], partial_data[2], x,y)

        return finished_data

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