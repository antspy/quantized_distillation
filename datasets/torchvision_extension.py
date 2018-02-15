#In this file some more transformations (apart from the ones defined in torchvision.transform)
#are added. Particularly helpful to train imagenet, and in the style of the transforms
#used by fb.resnet https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

#This file is taken from a proposed pull request on the torchvision github project.
#At the moment this pull request has not been accepted yet, that is why I report it here.
#Link to the pull request: https://github.com/pytorch/vision/pull/27/files

class Lighting(object):

    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        # img is supposed go be a torch tensor

        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
