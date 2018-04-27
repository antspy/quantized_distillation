#  Model compression via distillation and quantization

This code has been written to experiment with quantized distillation and differentiable quantization, techniques developed in our paper ["Model compression via distillation and quantization"](https://arxiv.org/abs/1802.05668).

If you find this code useful in your research, please cite the paper:

```
@article{2018arXiv180205668P,
   author = {{Polino}, A. and {Pascanu}, R. and {Alistarh}, D.},
    title = "{Model compression via distillation and quantization}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1802.05668},
 keywords = {Computer Science - Neural and Evolutionary Computing, Computer Science - Learning},
     year = 2018,
    month = feb,
}
```


The code is written in [Pytorch 0.3](http://pytorch.org/) using Python 3.6. It is not backward compatible with Python2.x

*Note* Pytorch 0.4 introduced some major breaking changes. To use this code, please use Pytorch 0.3.

# Getting started

### Prerequisites
This code is mostly self contained. Only a few additional libraries are requires, specified in [requirements.txt](requirements.txt). The repository already contains a fork of the [openNMT-py project](https://github.com/OpenNMT/OpenNMT-py). Note that, due to the rapidly changing nature of the openNMT-py codebase and the substantial time and effort required to make it compatible with our code, it is unlikely that we will support newer versions of openNMT-py.

### Summary of folder's content
This is a short explanation of the contents of each folder:

 - *datasets* is a package that automatically downloads and process several datasets, including CIFAR10, PennTreeBank, WMT2013, etc.
 - *quantization* contains the quantization functions that are used.
 - *perl_scripts* contains some perl scripts taken from the [moses project](https://github.com/moses-smt/mosesdecoder) to help with the translation task.
 - *onmt* contains the code from [openNMT-py project](https://github.com/OpenNMT/OpenNMT-py). It is slightly modified to make it consistent with our codebase.
 - *helpers* contains some functions used across the whole project.
 - *model_manager.py* contains a useful class that implements common I/O operations on saved models. It is especially useful when training multiple similar models, as it keeps track of the options with which the models were trained and the results of each training run. *Note*: it does not support concurrent access to the same files. I am working on a version that does; if you are interested, drop me a line.
 - First-level files like [cifar10_test.py](cifar10_test.py) are the main files that implement the experiments using the rest of the codebase.
 - Other folders contain model definitions and training routines, depending on the task.

### Running the code

The first thing to do is to import some dataset and create the train and test set loaders.
Define a folder where you want to save all your datasets; they will be automatically downloaded and processed in the folder specified. The following example shows how to load the CIFAR10 dataset, create and train a model.
```python
import dataset
dataset.BASE_DATA_FOLDER = '/home/saved_datasets'

batch_size = 50
cifar10 = datasets.CIFAR10() #-> will be saved in /home/saved_datasets/cifar10
train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)
```

Now we can use ```train_loader``` and ```test_loader``` as generators from which to get the train and test data as pytorch tensors.

At this point we just need to define a model and train it:

```python
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=True,
                                              useAffineTransformInBatchNorm=True)
convForwModel.train_model(teacherModel, train_loader, test_loader, epochs_to_train=200)
```

 As mentioned before, it is often better to use the ModelManager class to be able to automatically save the results and retrieve them later. So we would typically write

```python
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import model_manager
cifar10Manager = model_manager.ModelManager('model_manager_cifar10.tst',
                                            'model_manager', create_new_model_manager=False)#the first time set this to True
model_name = 'cifar10_teacher'
teacherModelPath = os.path.join(cifar10modelsFolder, model_name)
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=True,
                                              useAffineTransformInBatchNorm=True)
if not model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(model_name, teacherModelPath,
            arguments_creator_function={**convForwModel.teacherModelSpec,
                                        'useBatchNorm':True,
                                        'useAffineTransformInBatchNorm':True})
cifar10Manager.train_model(teacherModel, model_name=model_name,
                           train_function=convForwModel.train_model,
                           arguments_train_function={'epochs_to_train': 200},
                           train_loader=train_loader, test_loader=test_loader)
```         

This is the general structure necessary to use the code. For more examples, please look at one of the main files that are used to run the experiments.

# Authors

 - Antonio Polino
 - Razvan Pascanu
 - Dan Alistarh

# License

The code is licensed under the MIT Licence. See the [LICENSE.md](LICENSE.md) file for detail.

# Acknowledgements

We would like to thank Ce Zhang  (ETH Zürich), Hantian Zhang (ETH Zürich) and Martin Jaggi (EPFL) for their support with experiments and valuable feedback.
