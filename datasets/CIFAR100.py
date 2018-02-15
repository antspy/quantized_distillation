import os
import torchvision
import torchvision.transforms as vision_transforms
import datasets
import torch
import datasets.torchvision_extension as vision_transforms_extension


meanstd = {
   'mean': [0.5071, 0.4867, 0.4408],
   'std': [0.2675, 0.2565, 0.2761],
}

class CIFAR100(object):
    def __init__(self, dataFolder=None, pin_memory=False):

        self.dataFolder = dataFolder if dataFolder is not None else os.path.join(datasets.BASE_DATA_FOLDER, 'CIFAR100')
        self.pin_memory = pin_memory
        self.meanStd = meanstd

        #download the dataset
        torchvision.datasets.CIFAR100(self.dataFolder, download=True)

    def getTrainLoader(self, batch_size, shuffle=True, num_workers=1, checkFileIntegrity=False):

        #first we define the training transform we will apply to the dataset
        listOfTransoforms = []
        listOfTransoforms.append(vision_transforms.RandomCrop((32, 32), padding=4))
        listOfTransoforms.append(vision_transforms.RandomHorizontalFlip())
        # listOfTransoforms.append(vision_transforms.ColorJitter(brightness=0.4,
        #                                                                  contrast=0.4,
        #                                                                  saturation=0.4))
        listOfTransoforms.append(vision_transforms.ToTensor())
        listOfTransoforms.append(vision_transforms.Normalize(mean=self.meanStd['mean'],
                                                             std=self.meanStd['std']))

        train_transform = vision_transforms.Compose(listOfTransoforms)

        #define the trainset
        trainset = torchvision.datasets.CIFAR100(root=self.dataFolder, train=True,
                                        download=checkFileIntegrity, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=self.pin_memory)

        return trainloader

    def getTestLoader(self, batch_size, shuffle=True, num_workers=1, checkFileIntegrity=False):

        listOfTransoforms = [vision_transforms.ToTensor()]
        listOfTransoforms.append(vision_transforms.Normalize(mean=self.meanStd['mean'],
                                                             std=self.meanStd['std']))

        test_transform = vision_transforms.Compose(listOfTransoforms)

        testset = torchvision.datasets.CIFAR100(root=self.dataFolder, train=False,
                                               download=checkFileIntegrity, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers, pin_memory=self.pin_memory)

        return testloader