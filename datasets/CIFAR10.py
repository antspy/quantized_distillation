import os
import torchvision
import torchvision.transforms as vision_transforms
import datasets
import torch
import datasets.torchvision_extension as vision_transforms_extension
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    print('Cannot import matplotlib. CIFAR10.save_img method will crash if used')

meanstd = {
   'mean':[0.5, 0.5, 0.5],
   'std': [0.5, 0.5, 0.5],
}

class CIFAR10(object):
    def __init__(self, dataFolder=None, pin_memory=False):

        self.dataFolder = dataFolder if dataFolder is not None else os.path.join(datasets.BASE_DATA_FOLDER, 'CIFAR10')
        self.pin_memory = pin_memory
        self.meanStd = meanstd

        #download the dataset
        torchvision.datasets.CIFAR10(self.dataFolder, download=True)

        #add some useful metadata
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def getTrainLoader(self, batch_size, shuffle=True, num_workers=1, checkFileIntegrity=False):

        #first we define the training transform we will apply to the dataset
        listOfTransoforms = []
        listOfTransoforms.append(vision_transforms.RandomCrop((32, 32), padding=4))
        listOfTransoforms.append(vision_transforms.RandomHorizontalFlip())
        #
        # listOfTransoforms.append(vision_transforms.ColorJitter(brightness=0.4,
        #                                                                  contrast=0.4,
        #                                                                  saturation=0.4))
        listOfTransoforms.append(vision_transforms.ToTensor())
        # TODO: TO make this work I need the pca values, i.e. eigenvalues and eigenvectors
        # of the RGB colors, computed on a subset of the cifar10 dataset.
        # try to implement this at some point
        # listOfTransoforms.append(vision_transforms_extension.Lighting(alphastd=0.1,
        #                                                               eigval=self.pca['eigval'],
        #                                                               eigvec=self.pca['eigvec']))
        listOfTransoforms.append(vision_transforms.Normalize(mean=self.meanStd['mean'],
                                                             std=self.meanStd['std']))

        train_transform = vision_transforms.Compose(listOfTransoforms)

        #define the trainset
        trainset = torchvision.datasets.CIFAR10(root=self.dataFolder, train=True,
                                        download=checkFileIntegrity, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=self.pin_memory)

        return trainloader

    def getTestLoader(self, batch_size, shuffle=True, num_workers=1, checkFileIntegrity=False):

        listOfTransoforms = [vision_transforms.ToTensor()]
        listOfTransoforms.append(vision_transforms.Normalize(mean=self.meanStd['mean'],
                                                             std=self.meanStd['std']))

        test_transform = vision_transforms.Compose(listOfTransoforms)

        testset = torchvision.datasets.CIFAR10(root=self.dataFolder, train=False,
                                               download=checkFileIntegrity, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers, pin_memory=self.pin_memory)

        return testloader

    @staticmethod
    def save_img(img, path_file):
        try:
            img = img.data #in case a variable is passed
        except:pass
        mean_ = meanstd['mean']
        std_ = meanstd['std']
        meanDivStd = [-mean_[idx]/std_[idx] for idx in range(len(mean_))]
        inv_std = [1/std_[idx] for idx in range(len(std_))]
        img = vision_transforms.Normalize(meanDivStd, inv_std)(img)
        npimg = img.cpu().numpy()
        plt.imsave(path_file, np.transpose(npimg, (1, 2, 0)))