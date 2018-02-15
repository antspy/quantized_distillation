import os
import torch
import torchvision
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import datasets
import model_manager
from cnn_models.wide_resnet_imagenet import Wide_ResNet_imagenet
import cnn_models.resnet_kfilters as resnet_kfilters
import functools
import quantization
import helpers.functions as mhf

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {} GPUs'.format(cuda_devices, len(cuda_devices)))

print('Number of bits in training: {}'.format(4))

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'

USE_CUDA = torch.cuda.is_available()
NUM_GPUS = len(cuda_devices)

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

epochsToTrainImageNet = 90
imageNet12modelsFolder = os.path.join(SAVED_MODELS_FOLDER, 'imagenet12_new')
imagenet_manager = model_manager.ModelManager('model_manager_resnet34double.tst',
                                              'model_manager', create_new_model_manager=False)

for x in imagenet_manager.list_models():
    if imagenet_manager.get_num_training_runs(x) >= 1:
        s = '{}; Last prediction acc: {}, Best prediction acc: {}'.format(x,
                                            imagenet_manager.load_metadata(x)[1]['predictionAccuracy'][-1],
                                            max(imagenet_manager.load_metadata(x)[1]['predictionAccuracy']))
        print(s)

try:
    os.mkdir(imageNet12modelsFolder)
except:pass

TRAIN_QUANTIZED_DISTILLED = True

print('Batch size: {}'.format(batch_size))

if batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))


imageNet12 = datasets.ImageNet12('...',
                                 '...',
                                 type_of_data_augmentation='extended', already_scaled=False,
                                 pin_memory=True)


train_loader = imageNet12.getTrainLoader(batch_size, shuffle=True)
test_loader = imageNet12.getTestLoader(batch_size, shuffle=False)

# # Teacher model
resnet34 = torchvision.models.resnet34(True)  #already trained
if USE_CUDA:
    resnet34 = resnet34.cuda()
if NUM_GPUS > 1:
    resnet34 = torch.nn.parallel.DataParallel(resnet34)


#Train a wide-resNet with quantized distillation
quant_distilled_model_name = 'resnet18_1.5xfilters_quant_distilled4bits'
quantDistilledModelPath = os.path.join(imageNet12modelsFolder, quant_distilled_model_name)
quantDistilledOptions = {}
quant_distilled_model = resnet_kfilters.resnet18(k=1.5)

if USE_CUDA:
    quant_distilled_model = quant_distilled_model.cuda()
if NUM_GPUS > 1:
    quant_distilled_model = torch.nn.parallel.DataParallel(quant_distilled_model)

if not quant_distilled_model_name in imagenet_manager.saved_models:
    imagenet_manager.add_new_model(quant_distilled_model_name, quantDistilledModelPath,
                                   arguments_creator_function=quantDistilledOptions)

if TRAIN_QUANTIZED_DISTILLED:
    imagenet_manager.train_model(quant_distilled_model, model_name=quant_distilled_model_name,
                                 train_function=convForwModel.train_model,
                                 arguments_train_function={'epochs_to_train': epochsToTrainImageNet,
                                                           'learning_rate_style': 'imagenet',
                                                           'initial_learning_rate': 0.1,
                                                           'use_nesterov':True,
                                                           'initial_momentum':0.9,
                                                           'weight_decayL2':1e-4,
                                                           'start_epoch': 0,
                                                           'print_every':30,
                                                           'use_distillation_loss':True,
                                                           'teacher_model': resnet34,
                                                           'quantizeWeights':True,
                                                           'numBits':4,
                                                           'bucket_size':256,
                                                           'quantize_first_and_last_layer': False},
                                 train_loader=train_loader, test_loader=test_loader)
quant_distilled_model.load_state_dict(imagenet_manager.load_model_state_dict(quant_distilled_model_name))

# print(cnn_hf.evaluateModel(quant_distilled_model, test_loader, fastEvaluation=False))
# print(cnn_hf.evaluateModel(quant_distilled_model, test_loader, fastEvaluation=False, k=5))
# print(cnn_hf.evaluateModel(resnet34, test_loader, fastEvaluation=False))
# print(cnn_hf.evaluateModel(resnet34, test_loader, fastEvaluation=False, k=5))
# quant_fun = functools.partial(quantization.uniformQuantization, s=2**4, bucket_size=256)
# size_mb = mhf.get_size_quantized_model(quant_distilled_model, 4, quant_fun, 256,
#                                        quantizeFirstLastLayer=False)
# print(size_mb)
# print(mhf.getNumberOfParameters(quant_distilled_model)/1000000)
# print(mhf.getNumberOfParameters(resnet34) / 1000000)
