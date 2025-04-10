import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import time


class SingleSubjectDataset(Dataset):
    """Dataset for a single timeseries. This is not used as a standalone."""
    def __init__(self, data, seq_len, size, device="cpu"):
        """Initializes the dataset.
        Args:
            data: data tensor. Expected to be 2D tensor (T x dz)
            seq_len: length of the sequences during training
            size: size of the data slice that is used for training
            device: device to store the data on"""
        # load data
        self.data = data
        # store necessary variables
        self.seq_len = seq_len
        self.size = size
        # if size is larger than the data, set it to the data size minus one sequence length for testing
        if self.size > self.data.shape[0]:
            self.size = self.data.shape[0] - self.seq_len - 1
        else:
            self.size = self.size - self.seq_len - 1
        # create start indices for train sequences
        self.indices = np.arange(self.size)
        # store device
        self.device = device
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # get slices for data and target
        data_slice = slice(self.indices[idx], self.indices[idx] + self.seq_len)
        target_slice = slice(self.indices[idx] + 1, self.indices[idx] + self.seq_len + 1)
        # move to device
        return self.data[data_slice], self.data[target_slice]
    
    def get_data(self):
        """Returns the raw data. Both train and test portion."""
        return self.data

    def get_test_data(self):
        """Returns the test set."""
        return self.data[self.size:].to(self.device)


class MultiSubjectDataset(Dataset):
    """Main dataset class. Consists of one or multiple single subject datasets"""
    def __init__(self, path, seq_len, size, subjects_per_batch, num_workers, device="cpu"):
        """Initializes the dataset.
        Args:
            path: full path to the data. Expected to be .pt file
                containing a timeseries for each subject 
                (-> shape: num_subjects x num_timesteps x num_features).
            seq_len: length of the sequences during training
            size: size of the data slice that is used for training
            subjects_per_batch: number of subjects to sample from each sequence
                in the batch
            num_workers: number of workers for the dataloader
            device: device to store the data on
        """
        self.num_workers = num_workers
        self.data = torch.load(path)
        # add subject dimension for single subject datasets
        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(0)
        self.datasets = []
        for i in range(self.data.shape[0]):
            self.datasets.append(SingleSubjectDataset(self.data[i], seq_len, size, device=device))
        self.num_subjects = len(self.datasets)
        self.subjects_per_batch = subjects_per_batch
        if self.subjects_per_batch is None or self.subjects_per_batch > self.num_subjects:
            self.subjects_per_batch = self.num_subjects
        self.shuffle_subjects()

    def __len__(self):
        return len(self.env_indices)
    
    def __getitem__(self, idx):
        return self.datasets[self.env_indices[idx]][self.sample_indices[idx]] + (self.env_indices[idx], )
    
    def shuffle_subjects(self):
        """Shuffles the subjects from which the data is sampled."""
        sample_from = np.random.choice(self.num_subjects, self.subjects_per_batch, replace=False)
        self.env_indices = np.concat([i * np.ones(len(self.datasets[i])) for i in sample_from]).astype(int)
        self.sample_indices = np.concat([np.arange(len(self.datasets[i])) for i in sample_from]).astype(int)
    
    def get_data(self):
        """Returns the raw data. Both train and test portions."""
        return self.data
    
    def get_test_data(self):
        """Returns the test sets of the sub data sets."""
        return torch.stack([ds.get_test_data() for ds in self.datasets], dim=0)
    
    def determine_num_workers(self, batch_size, bpe):
        sampler = RandomSampler(self, num_samples=bpe*batch_size, replacement=bpe*batch_size > len(self))
        times = []
        nums = range(13)
        for nw in nums:
            start = time.time()
            for _ in DataLoader(self, batch_size, sampler=sampler, num_workers=nw):
                pass
            times.append(time.time() - start)
        self.num_workers = nums[np.argmin(times)]

    def get_dataloader(self, batch_size, bpe):
        """Returns a dataloader with the specified number of workers. If None is passed
        the optimal number of workers is determined, which can be time consuming for 
        larger datasets."""
        self.shuffle_subjects()
        if self.num_workers is None:
            self.determine_num_workers(batch_size, bpe)

        sampler = RandomSampler(self, num_samples=bpe*batch_size, replacement=bpe*batch_size > len(self))
        return DataLoader(self, batch_size, sampler=sampler, num_workers=self.num_workers)