import torch
import torchvision
import torchvision.transforms as vision_transforms
import torch.utils.data
import datasets.torchvision_extension as vision_transforms_extension

#For this dataset, automatic download has not been implemented. You have to provide path to the train and test folders
#formatted as described here: http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
#Essentially images in the same class must be in the same folder with the name of the class, like so:
# root/dog/xxx.png
# root/dog/xxy.png
# root/dog/xxz.png
#
# root/cat/123.png
# root/cat/nsdf3.png
# root/cat/asd932_.png

#To prepare the imagenet2012 dataset in such a way, follow the instructions at
# https://github.com/soumith/imagenet-multiGPU.torch
# It says:
# The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968).
# You need to get the validation groundtruth and move the validation images into appropriate subfolders.
# To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands:
# extract train data
# mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
# tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# # extract validation data
# cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

#Now you're all set!


#Computed from random subset of ImageNet training images
#This values were taken from the fb.resnet github project linked above
meanstd = {
   'mean':[0.485, 0.456, 0.406],
   'std': [0.229, 0.224, 0.225],
}

pca = {
   'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
   'eigvec': torch.Tensor([
      [-0.5675,  0.7192,  0.4009],
      [-0.5808, -0.0045, -0.8140],
      [-0.5836, -0.6948,  0.4203],
   ])
}

class ImageNet12(object):
    def __init__(self, trainFolder, testFolder, pin_memory=False, size_images=224,
                 scaled_size=256, type_of_data_augmentation='basic', already_scaled=False):
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.pin_memory = pin_memory
        self.meanstd = meanstd
        self.pca = pca
        #images will be rescaled to match this size
        if not isinstance(size_images, int):
            raise ValueError('size_images must be an int. It will be scaled to a square image')
        self.size_images = size_images
        self.scaled_size = scaled_size
        type_of_data_augmentation = type_of_data_augmentation.lower()
        if type_of_data_augmentation not in ('basic', 'extended'):
            raise ValueError('type_of_data_augmentation must be either basic or extended')
        self.type_of_data_augmentation = type_of_data_augmentation
        self.already_scaled = already_scaled # if you scaled all the images before training (see link above)
                                             # then set this to True

    def getTrainLoader(self, batch_size, shuffle=True, num_workers=4):

        # first we define the training transform we will apply to the dataset
        list_of_transforms = []
        list_of_transforms.append(vision_transforms.RandomSizedCrop(self.size_images))
        list_of_transforms.append(vision_transforms.RandomHorizontalFlip())

        if self.type_of_data_augmentation == 'extended':
            list_of_transforms.append(vision_transforms.ColorJitter(brightness=0.4,
                                                                             contrast=0.4,
                                                                             saturation=0.4))
        list_of_transforms.append(vision_transforms.ToTensor())
        if self.type_of_data_augmentation == 'extended':
            list_of_transforms.append(vision_transforms_extension.Lighting(alphastd=0.1,
                                                                          eigval=self.pca['eigval'],
                                                                          eigvec=self.pca['eigvec']))

        list_of_transforms.append(vision_transforms.Normalize(mean=self.meanstd['mean'],
                                                             std=self.meanstd['std']))
        train_transform = vision_transforms.Compose(list_of_transforms)
        train_set = torchvision.datasets.ImageFolder(self.trainFolder, train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=num_workers, pin_memory=self.pin_memory)

        return train_loader

    def getTestLoader(self, batch_size, shuffle=True, num_workers=4):
        # first we define the training transform we will apply to the dataset
        list_of_transforms = []
        if self.already_scaled is False:
            list_of_transforms.append(vision_transforms.Resize(self.scaled_size))
        list_of_transforms.append(vision_transforms.CenterCrop(self.size_images))
        list_of_transforms.append(vision_transforms.ToTensor())
        list_of_transforms.append(vision_transforms.Normalize(mean=self.meanstd['mean'],
                                                             std=self.meanstd['std']))

        test_transform = vision_transforms.Compose(list_of_transforms)

        test_set = torchvision.datasets.ImageFolder(self.testFolder, test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=num_workers, pin_memory=self.pin_memory)

        return test_loader