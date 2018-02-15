import model_manager
import torch
import os
import datasets
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import quantization
import pickle
import copy
import quantization.help_functions as qhf
import functools
import helpers.functions as mhf
import itertools as it

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'
USE_CUDA = torch.cuda.is_available()

print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass


cifar10Manager = model_manager.ModelManager('model_manager_cifar10.tst',
                                            'model_manager', create_new_model_manager=False)
cifar10modelsFolder = os.path.join(SAVED_MODELS_FOLDER, 'cifar10')

for x in cifar10Manager.list_models():
    if cifar10Manager.get_num_training_runs(x) >= 1:
        print(x, cifar10Manager.load_metadata(x)[1]['predictionAccuracy'][-1])

try:
    os.mkdir(cifar10modelsFolder)
except:pass

USE_BATCH_NORM = True
AFFINE_BATCH_NORM = True


COMPUTE_DIFFERENT_HEURISTICS = False

batch_size = 25
cifar10 = datasets.CIFAR10()
train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)


## distilled model
distilled_model_name = 'cifar10_distilled'
distilledModelSpec = copy.deepcopy(convForwModel.smallerModelSpec)
distilledModelSpec['spec_dropout_rates'] = [] #no dropout with distilled model


def values_param_iter(n=3):
    return it.product(*([True, False],)*n)

numBits = [2, 4]
if COMPUTE_DIFFERENT_HEURISTICS:
    for numBit in numBits:
        for assign_bits_auto, use_distillation_loss, compute_initial_points in values_param_iter(n=3):
            if compute_initial_points is True:
                compute_initial_points = 'quantiles'
            else:
                compute_initial_points = 'uniform'
            str_identifier = 'quantpoints{}bits_auto{}_distill{}_initial"{}"'.format(numBit, assign_bits_auto,
                                                                                  use_distillation_loss,
                                                                                  compute_initial_points)
            distilled_quantized_model_name = distilled_model_name + str_identifier
            distilled_quantized_model = convForwModel.ConvolForwardNet(**distilledModelSpec,
                                                                       useBatchNorm=USE_BATCH_NORM,
                                                                       useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
            if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
            distilled_quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(distilled_model_name))
            epochs_to_train = 50

            quantized_model_dict, quantization_points, infoDict = convForwModel.optimize_quantization_points(
                distilled_quantized_model,
                train_loader, test_loader, numPointsPerTensor=2**numBit,
                assignBitsAutomatically=assign_bits_auto,
                bucket_size=256, epochs_to_train=epochs_to_train,
                use_distillation_loss=use_distillation_loss, initial_learning_rate=1e-5,
                initialize_method=compute_initial_points)
            quantization_points = [x.data.view(1,-1).cpu().numpy().tolist()[0] for x in quantization_points]
            save_path = cifar10Manager.get_model_base_path(distilled_model_name) + str_identifier
            with open(save_path, 'wb') as p:
                pickle.dump((quantization_points, infoDict), p)
            torch.save(quantized_model_dict, save_path + '_model_state_dict')

for numBit in numBits:
    for assign_bits_auto, use_distillation_loss, compute_initial_points in values_param_iter(n=3):
        if compute_initial_points is True:
            compute_initial_points = 'quantiles'
        else:
            compute_initial_points = 'uniform'
        str_identifier = 'quantpoints{}bits_auto{}_distill{}_initial"{}"'.format(numBit, assign_bits_auto,
                                                                                 use_distillation_loss,
                                                                                 compute_initial_points)
        distilled_quantized_model_name = distilled_model_name + str_identifier
        save_path = cifar10Manager.get_model_base_path(distilled_model_name) + str_identifier
        with open(save_path, 'rb') as p:
            quantization_points, infoDict = pickle.load(p)

        distilled_quantized_model = convForwModel.ConvolForwardNet(**distilledModelSpec,
                                                                   useBatchNorm=USE_BATCH_NORM,
                                                                   useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
        if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
        distilled_quantized_model.load_state_dict(torch.load(save_path + '_model_state_dict'))
        reported_accuracy = max(infoDict['predictionAccuracy'])
        actual_accuracy = cnn_hf.evaluateModel(distilled_quantized_model, test_loader) #this corresponds to the last one
        #the only problem is that I don't save the model with the max accuracy, but the model at the last epoch
        print('Model "{}" => reported accuracy: {} - actual accuracy: {}'.format(distilled_quantized_model_name,
                                                                                 reported_accuracy, actual_accuracy))
