from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torch

class CIFAR100C(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"  #"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-100-c-python.tar"
    tgz_md5 = '11f0ed0f1191edbf9fa23466ae6021d3'
   

    def __init__(
            self,
            root: str,
            train: bool = True,
            dgrd = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR100C, self).__init__(root, transform=transform,
                                      target_transform=target_transform)


        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        self.targets = np.load(self.root+'/CIFAR-100-C/labels.npy')
        self.data = np.load(self.root +'/CIFAR-100-C/' +dgrd['type'] +'.npy')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)       
        # import ipdb;ipdb.set_trace()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img.type(torch.float), target.astype(np.int)


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        fpath = os.path.join(self.root, self.filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


