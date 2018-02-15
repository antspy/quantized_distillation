import os
import torch
import torchvision
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import datasets
import model_manager

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {} GPUs'.format(cuda_devices, len(cuda_devices)))


if 'NUM_BITS' in os.environ:
    NUM_BITS = int(os.environ['NUM_BITS'])
else:
    NUM_BITS = 4

print('Number of bits in training: {}'.format(NUM_BITS))

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
imagenet_manager = model_manager.ModelManager('model_manager_imagenet_distilled_New{}bits.tst'.format(NUM_BITS),
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
# resnet152 = torchvision.models.resnet152(True)  #already trained
# if USE_CUDA:
#     resnet152 = resnet152.cuda()
# if NUM_GPUS > 1:
#     resnet152 = torch.nn.parallel.DataParallel(resnet152)


#normal resnet18 training
resnet18 = torchvision.models.resnet18(False) #not pre-trained, 11.7 million parameters
if USE_CUDA:
    resnet18 = resnet18.cuda()
if NUM_GPUS > 1:
    resnet18 = torch.nn.parallel.DataParallel(resnet18)
model_name = 'resnet18_normal_fullprecision'
model_path = os.path.join(imageNet12modelsFolder, model_name)

if not model_name in imagenet_manager.saved_models:
    imagenet_manager.add_new_model(model_name, model_path,
                                   arguments_creator_function={'loaded_from':'torchvision_models'})

imagenet_manager.train_model(resnet18, model_name=model_name,
                             train_function=convForwModel.train_model,
                             arguments_train_function={'epochs_to_train': epochsToTrainImageNet,
                                                       'learning_rate_style': 'imagenet',
                                                       'initial_learning_rate': 0.1,
                                                       'weight_decayL2':1e-4,
                                                       'start_epoch':0,
                                                       'print_every':30},
                             train_loader=train_loader, test_loader=test_loader)

#distilled
# resnet18_distilled = torchvision.models.resnet18(False) #not pre-trained, 11.7 million parameters
# if USE_CUDA:
#     resnet18_distilled = resnet18_distilled.cuda()
# if NUM_GPUS > 1:
#     resnet18_distilled = torch.nn.parallel.DataParallel(resnet18_distilled)
# model_name = 'resnet18_distilled'
# model_path = os.path.join(imageNet12modelsFolder, model_name)
#
# if not model_name in imagenet_manager.saved_models:
#     imagenet_manager.add_new_model(model_name, model_path,
#                                    arguments_creator_function={'loaded_from':'torchvision_models'})

# imagenet_manager.train_model(resnet18_distilled, model_name=model_name,
#                              train_function=convForwModel.train_model,
#                              arguments_train_function={'epochs_to_train': epochsToTrainImageNet,
#                                                        'teacher_model': resnet34,
#                                                        'learning_rate_style': 'imagenet',
#                                                        'initial_learning_rate': initial_lr,
#                                                        'weight_decayL2':1e-4,
#                                                        'use_distillation_loss':True,
#                                                        'start_epoch':start_epoch,
#                                                        'print_every':100},
#                              train_loader=train_loader, test_loader=test_loader)

#quantized distilled
# bits_to_try = [NUM_BITS]
#
# for numBit in bits_to_try:
#     resnet18_quant_distilled = torchvision.models.resnet18(False) #not pre-trained, 11.7 million parameters
#     if USE_CUDA:
#         resnet18_quant_distilled = resnet18_quant_distilled.cuda()
#     if NUM_GPUS > 1:
#         resnet18_quant_distilled = torch.nn.parallel.DataParallel(resnet18_quant_distilled)
#     model_name = 'resnet18_quant_distilled_{}bits'.format(numBit)
#     model_path = os.path.join(imageNet12modelsFolder, model_name)
#
#     if not model_name in imagenet_manager.saved_models:
#         imagenet_manager.add_new_model(model_name, model_path,
#                                        arguments_creator_function={'loaded_from':'torchvision_models'})
#
#     imagenet_manager.train_model(resnet18_quant_distilled, model_name=model_name,
#                                  train_function=convForwModel.train_model,
#                                  arguments_train_function={'epochs_to_train': epochsToTrainImageNet,
#                                                            'learning_rate_style': 'imagenet',
#                                                            'initial_learning_rate': 0.1,
#                                                            'use_nesterov':True,
#                                                            'initial_momentum':0.9,
#                                                            'weight_decayL2':1e-4,
#                                                            'start_epoch': 0,
#                                                            'print_every':30,
#                                                            'use_distillation_loss':True,
#                                                            'teacher_model': resnet152,
#                                                            'quantizeWeights':True,
#                                                            'numBits':numBit,
#                                                            'bucket_size':256,
#                                                            'quantize_first_and_last_layer': False},
#                                  train_loader=train_loader, test_loader=test_loader)
