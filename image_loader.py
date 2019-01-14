"""
Loading and using the Mini-ImageNet dataset.

To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import copy
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from supervised_reptile.util import list_dir, list_files

# Default transforms
transform_image = transforms.Compose([
    transforms.ToTensor()
])

def read_image(path, rotation=0):
    img = Image.open(path, mode='r').convert('RGB')
#    img = Image.open(path, mode='r').convert('L')
    img = img.resize((80, 80)).rotate(rotation)
    img = transform_image(img)
    return img


class FewShot(Dataset):
    '''
    Dataset for K-shot N-way classification
    '''
    def __init__(self, samples, parent=None):
        self.samples = samples
        self.parent = parent

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]['path']
        name = self.samples[idx]['name']
        rotation = self.samples[idx]['rotation']
        if name in self.parent._cache:
            image = self.parent._cache[name]
        else:
            image = read_image(path, rotation)
            self.parent._cache[name] = image

        label = self.samples[idx]['batch_idx']
        return image, label

class get_task:   
    def get_random_task_split(self, num_classes=5, train_shots=10, test_shots=0):
        train_samples = []
        test_samples = []
        sample_indices = np.random.choice(len(self), num_classes, replace=False)
        for batch_idx, idx in enumerate(sample_indices):
            class_dir, paths = self.class_list[idx]
            for i, path in enumerate(np.random.choice(paths, train_shots + test_shots, replace=False)):
                new_path = {}
                new_path.update(path)
                new_path['batch_idx'] = batch_idx
                if i < train_shots:
                    train_samples.append(new_path)
                else:
                    test_samples.append(new_path)
        train_task = FewShot(train_samples, parent=self)
        test_task = FewShot(test_samples, parent=self)
        return train_task, test_task

class MiniimagenetFolder(get_task):
    
    def __init__(self, dir_path):
        self._cache = {}
        _clases = {}
        # Open and load text file including the whole training data
        for subfolder in list_dir(dir_path):
            class_dir = os.path.join(dir_path, subfolder)
            class_idx = len(_clases)
            _clases[class_dir] = []
            for filename in list_files(class_dir, '.JPEG'): #[f for f in os.listdir(class_dir) if f.endswith('.JPEG')]
                _clases[class_dir].append({
                    'path': os.path.join(class_dir, filename),
                    'class_idx': class_idx,
                    'name': os.path.splitext(filename)[0],
                    'rotation': 0
                })
        self.class_list = list(_clases.items())
        
    def __len__(self):
        return len(self.class_list)    

def read_dataset(data_dir):
    """
    Read the Mini-ImageNet dataset.

    Args:
      data_dir: directory containing Mini-ImageNet.

    Returns:
      A tuple (train, val, test) of sequences of
        ImageNetClass instances.
    """
    return tuple(MiniimagenetFolder(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])   


class OmniglotFolder(get_task):

    def __init__(self, dir_path):
        self._cache = {}
        _clases = {}
        for subfolder in list_dir(dir_path):
            for character in list_dir(os.path.join(dir_path, subfolder)):
                class_dir = os.path.join(dir_path, subfolder, character)
                class_idx = len(_clases)
                _clases[class_dir] = []
                for filename in list_files(class_dir, '.png'):
                    _clases[class_dir].append({
                        'path': os.path.join(class_dir, filename),
                        'class_idx': class_idx,
                        'name': character + '_' + os.path.splitext(filename)[0],
                        'rotation': 0
                    })
        self.class_list = list(_clases.items())

    def __len__(self):
        return len(self.class_list)   

def SplitClasses(omniglot, validation=0.1):
    '''
    Split meta-omniglot into two meta-datasets of tasks (disjoint characters)
    '''
    n_val = int(validation * len(omniglot))
    indices = np.arange(len(omniglot))
    np.random.shuffle(indices)
    
    train_set = omniglot
    test_set = copy.deepcopy(omniglot)
    train_set.class_list = [train_set.class_list[i] for i in indices[:-n_val]]
    test_set.class_list = [test_set.class_list[i] for i in indices[-n_val:]]
    
    return train_set, test_set

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.
    """
    char_list = dataset.class_list
    for rotation in [90, 180, 270]:
        list_copy = copy.deepcopy(char_list)
        for i in range(len(list_copy)):
            for j in range(len(list_copy[i][1])):
                list_copy[i][1][j]['rotation'] = rotation
                list_copy[i][1][j]['name'] = list_copy[i][1][j]['name'] + '_' + str(rotation)   
        dataset.class_list = dataset.class_list + list_copy
        
    return dataset