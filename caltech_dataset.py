import torchvision
from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def make_dataset(
    directory,
    split,
    class_to_idx,
):
    instances = []
    directory = os.path.expanduser(directory)
    split = os.path.abspath(os.path.join(directory, os.pardir)) +"/" +split + ".txt"
    with open(split) as file_in:
        lines = []
        for line in file_in:
            lines.append(line)
    
    for line in lines:
        target_class = line.split("/")[0]
        if(target_class == "BACKGROUND_Google"):
            continue

        class_index = class_to_idx[target_class]
        #print("target_class = ", target_class, "/t index = ", class_index)

        #target_dir = os.path.join(directory, target_class)
        path = os.path.join(directory, line)
        if path.lower()[:len(path) -1].endswith("jpg") or path.lower()[:len(path) -1].endswith("jpeg") :
            item = path[:len(path) -1], class_index
            instances.append(item)

    return instances


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, split, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.extensions = IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = []
        for d in os.scandir(dir):
          if (d.is_dir() and ""+d.name != "BACKGROUND_Google"):
            classes.append(d.name)

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        path, label =  self.samples[index]  # Provided a way to access image and label via index
                                            # Image should be a PIL Image
                                            # label can be int

        image = pil_loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length