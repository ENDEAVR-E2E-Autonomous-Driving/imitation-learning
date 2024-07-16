import os
import h5py
import numpy as np
import time

from torch.utils.data import Dataset


class DataManager:
    def __init__(self, folder, name):
        self.folder = folder
        self.name = name
        self.file_path = self.get_file_path()
        self.h5file = None
        self.initialize_folder()
        self.open_file()

    def initialize_folder(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_file_path(self):
        return os.path.join(self.folder, f"{self.name}_{int(time.time())}.h5")

    def open_file(self):
        self.h5file = h5py.File(self.file_path, 'a')
        self.initialize_datasets()

    def initialize_datasets(self):
        datasets = {
            'images': ((0, 200, 200, 3), np.float32),
            'scalars': ((0, 1), np.float32),
            'targets': ((0, 3), np.float32),
        }
        for name, (shape, dtype) in datasets.items():
            if name not in self.h5file:
                self.h5file.create_dataset(name, shape, maxshape=(None, *shape[1:]), dtype=dtype)

    def save(self, image, scalars, targets):
        image = np.array(image, dtype=np.float32)
        scalars = np.array(scalars, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        try:
            for dataset_name, data in zip(['images', 'scalars', 'targets'], [image, scalars, targets]):
                dataset = self.h5file[dataset_name]
                dataset.resize(dataset.shape[0] + 1, axis=0)
                dataset[-1:] = data
        except Exception as e:
            print(f"An error occurred: {e}")

class ImitationDataset(Dataset):
    def __init__(self, folder):
        self.file_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')])
        self.file_lengths = [self.get_file_length(f) for f in self.file_paths]
        self.cumulative_lengths = np.cumsum(self.file_lengths)

    def get_file_length(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return file['images'].shape[0]

    def __len__(self):
        return sum(self.file_lengths)

    def get_file_for_index(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        file_path = self.file_paths[file_index]

        return file_path.split("/")[-1]
    
    def get_start_index(self, idx):
        # return 0 for first file, else return the index of the last element in the previous file
        if idx == 0:
            return 0
        else:
            return self.cumulative_lengths[idx - 1]
            
    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if file_index == 0:
            local_index = idx
        else:
            local_index = idx - self.cumulative_lengths[file_index - 1]

        file_path = self.file_paths[file_index]

        with h5py.File(file_path, 'r') as file:
            image = file['images'][local_index]
            scalars = file['scalars'][local_index]
            targets = file['targets'][local_index]

        image = np.array(image, dtype=np.float32).transpose((2, 0, 1))
        scalars = np.array(scalars, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return image, scalars, targets
    
    # removes files in a range of indices across multiple files (inclusive of start_idx, exclusive of end_idx)
    # returns -1 if the range is invalid, 0 otherwise
    def delete_groups(self, start_idx, end_idx):
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(self):
            return -1
        
        start_file_index = np.searchsorted(self.cumulative_lengths, start_idx, side='right')
        end_file_index = np.searchsorted(self.cumulative_lengths, end_idx, side='right')

        for file_index in range(start_file_index, end_file_index + 1):
            file_path = self.file_paths[file_index]

            with h5py.File(file_path, 'a') as file:
                # adapted from https://stackoverflow.com/questions/50222486/can-i-delete-an-element-from-an-hdf5-dataset
                local_start_index = start_idx - self.get_start_index(file_index)
                local_end_index = end_idx - self.get_start_index(file_index)
                
                if local_end_index > self.file_lengths[file_index]:
                    local_end_index = self.file_lengths[file_index]

                images = file['images']
                scalars = file['scalars']
                targets = file['targets']

                new_images = np.delete(images, slice(local_start_index, local_end_index), axis=0)
                new_scalars = np.delete(scalars, slice(local_start_index, local_end_index), axis=0)
                new_targets = np.delete(targets, slice(local_start_index, local_end_index), axis=0)

                del file['images']
                del file['scalars']
                del file['targets']

                file.create_dataset('images', data=new_images)
                file.create_dataset('scalars', data=new_scalars)
                file.create_dataset('targets', data=new_targets)
                
            start_idx = self.cumulative_lengths[file_index]
    
        self.file_lengths = [self.get_file_length(f) for f in self.file_paths]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.__len__()
        return 0
