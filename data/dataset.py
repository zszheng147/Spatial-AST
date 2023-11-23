import csv
import json
import math
import os
import random
import sys

import h5py
import numpy as np
import soundfile as sf
from scipy import signal

import torch
import torch.distributed as dist
import torch.nn.functional
import torchaudio
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler

class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        
class DistributedWeightedSampler(Sampler):
    #dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class MultichannelDataset(Dataset):
    _ext_reverb = ".npy"
    _ext_audio = ".wav"
    audio_path_root = "/mnt/lustre/sjtu/shared/data/raa/AudioSet"
    reverb_path_root = "/data/shared/zsz01/reverb/"

    def __init__(
            self, 
            audio_json, audio_conf,
            reverb_json, reverb_type,  
            label_csv=None,
            use_fbank=False, 
            fbank_dir=None, 
            roll_mag_aug=False, 
            mode='train'
        ):

        self.data = json.load(open(audio_json, 'r'))

        self.reverb = json.load(open(reverb_json, 'r'))['data']
        self.reverb_type = reverb_type
        self.channel_num = 2 if reverb_type == 'BINAURAL' else 9 if reverb_type == 'AMBISONICS' else 1
        
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.mixup = self.audio_conf.get('mixup')
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.noise = self.audio_conf.get('noise')
        
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.roll_mag_aug = roll_mag_aug

        self.mode = mode
        print(f'multilabel: {self.multilabel}')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        print('using mix-up with rate {:f}'.format(self.mixup))
        if self.noise == True:
            print('now use noise augmentation')
        print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset: {self.__len__()}')

    def _roll_mag_aug(self, waveform):
        idx = np.random.randint(len(waveform))
        mag = np.random.beta(10, 10) + 0.5
        return torch.roll(waveform, idx) * mag

    def _spatial_targets_discretize(self, reverb_item):
        distance = reverb_item['distance'] # meter: [0, 1, 2, 3, ..., 10]
        direction = reverb_item['direction']
        azimuth = reverb_item['azimuth_degrees'] # (-180, 180) = [-179~179]
        elevation = reverb_item['elevation_degrees'] # (-90, 90) = [-89~89]

        distance = distance # 11 classes
        azimuth = azimuth + 180
        elevation = elevation + 90
        # azimuth = (round(azimuth / 5.0) * 5 + 180) / 5.0 % 72
        # elevation = (round(elevation / 5.0) * 5 + 90) / 5.0 % 36

        spaital_targets = { 
            "distance": distance,         
            "direction": direction,
            "azimuth": azimuth,
            "elevation": elevation        
        }   
        return spaital_targets

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            reverb_item = random.choice(self.reverb)
            spaital_targets = self._spatial_targets_discretize(reverb_item)

            house_id, prefix  = reverb_item['reverberation'].split('-', maxsplit=1)
            reverb_path = os.path.join(self.reverb_path_root, self.reverb_type, house_id, prefix + self._ext_reverb)
            reverb = torch.from_numpy(np.load(reverb_path)).float()
            reverb_padding = 32000 * 2 - reverb.shape[1]
            if reverb_padding > 0:
                reverb = torch.nn.functional.pad(reverb, (0, reverb_padding), 'constant', 0)
            elif reverb_padding < 0:
                reverb = reverb[:, :32000 * 2]

            datum = self.data[index]

            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            
            audio_path = os.path.join(self.audio_path_root, datum['folder'], datum['id'] + self._ext_audio)
            mix_audio_path = os.path.join(self.audio_path_root, mix_datum['folder'], mix_datum['id'] + self._ext_audio)
            if 'unbalanced' in audio_path:
                h5_path, fname = audio_path.rsplit('/', 1)
                waveform = h5py.File(h5_path, "r")[fname][:].astype(np.float32)
                sr = 32000
            else:
                waveform, sr = sf.read(audio_path)
            if len(waveform.shape) > 1: 
                waveform = waveform[:, 0]  
            if sr != 32000:
                waveform = signal.resample_poly(waveform, 32000, sr)

            if 'unbalanced' in mix_audio_path:
                h5_path, fname = mix_audio_path.rsplit('/', 1)
                mix_waveform = h5py.File(h5_path, "r")[fname][:].astype(np.float32)
                sr = 32000
            else:
                mix_waveform, sr = sf.read(mix_audio_path)
            if len(mix_waveform.shape) > 1: 
                mix_waveform = mix_waveform[:, 0]  
            if sr != 32000:
                mix_waveform = signal.resample_poly(mix_waveform, 32000, sr)
       
            waveform = torch.from_numpy(waveform).reshape(1, -1).float()
            mix_waveform = torch.from_numpy(mix_waveform).reshape(1, -1).float()

            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
                mix_waveform = self._roll_mag_aug(mix_waveform)

            padding = 32000 * 10 - waveform.shape[1]
            if padding > 0:
                waveform = torch.nn.functional.pad(waveform, (0, padding), 'constant', 0)
            elif padding < 0:
                waveform = waveform[:, :32000 * 10]

            mix_padding = 32000 * 10 - mix_waveform.shape[1]
            if mix_padding > 0:
                mix_waveform = torch.nn.functional.pad(mix_waveform, (0, mix_padding), 'constant', 0)
            elif mix_padding < 0:
                mix_waveform = mix_waveform[:, :32000 * 10]
            
            mix_lambda = np.random.beta(10, 10)
            waveform = mix_lambda * waveform + (1 - mix_lambda) * mix_waveform

            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['label']:
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['label']:
                label_indices[int(self.index_dict[label_str])] += 1.0 - mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        
        else:
            datum = self.data[index]
            reverb_item = random.choice(self.reverb)

            label_indices = np.zeros(self.label_num)
            
            audio_path = os.path.join(self.audio_path_root, datum['folder'], datum['id'] + self._ext_audio)
            house_id, prefix  = reverb_item['reverberation'].split('-', maxsplit=1)
            reverb_path = os.path.join(self.reverb_path_root, self.reverb_type, house_id, prefix + self._ext_reverb)
            reverb = torch.from_numpy(np.load(reverb_path)).float()

            reverb_padding = 32000 * 2 - reverb.shape[1]
            if reverb_padding > 0:
                reverb = torch.nn.functional.pad(reverb, (0, reverb_padding), 'constant', 0)
            elif reverb_padding < 0:
                reverb = reverb[:, :32000 * 2]

            if 'unbalanced' in audio_path:
                h5_path, fname = audio_path.rsplit('/', 1)
                waveform = h5py.File(h5_path, "r")[fname][:].astype(np.float32)
                sr = 32000
            else:
                waveform, sr = sf.read(audio_path)
            if len(waveform.shape) > 1: 
                waveform = waveform[:, 0]   
           
            if sr != 32000:
                waveform = signal.resample_poly(waveform, 32000, sr)

            waveform = torch.from_numpy(waveform).reshape(1, -1).float()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)

            padding = 32000 * 10 - waveform.shape[1]
            if padding > 0:
                waveform = torch.nn.functional.pad(waveform, (0, padding), 'constant', 0)
            elif padding < 0:
                waveform = waveform[:, :32000 * 10]

            for label_str in datum['label']:
                label_indices[int(self.index_dict[label_str])] = 1.0
            label_indices = torch.FloatTensor(label_indices)

            spaital_targets = self._spatial_targets_discretize(reverb_item)

        return waveform, reverb, label_indices, spaital_targets, audio_path, reverb_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        waveforms, reverbs, label_indices, spaital_targets, audio_path, reverb_path = zip(*batch)
        waveforms = torch.stack(waveforms)
        
        reverbs = torch.stack(reverbs)
        # reverbs = [x.transpose(0, 1) for x in reverbs]

        # reverbs = pad_sequence(reverbs, batch_first=True, padding_value=0).transpose(1, 2)

        # spaital_targets
        spatial_targets_dict = {}
        spatial_targets_dict['distance'] = torch.LongTensor([x['distance'] for x in spaital_targets])
        spatial_targets_dict['azimuth'] = torch.LongTensor([x['azimuth'] for x in spaital_targets])
        spatial_targets_dict['elevation'] = torch.LongTensor([x['elevation'] for x in spaital_targets])

        return waveforms, reverbs, torch.stack(label_indices), spatial_targets_dict, audio_path, reverb_path