import os
import torch
import torchvision
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import datasets
import model_manager

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {} GPUs'.format(cuda_devices, len(cuda_devices)))


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
imagenet_manager = model_manager.ModelManager('model_manager_imagenet_Alexnet_distilled4bits.tst',
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
alexnet_unquantized = torchvision.models.alexnet(pretrained=True)
if USE_CUDA:
    alexnet_unquantized = alexnet_unquantized.cuda()
if NUM_GPUS > 1:
    alexnet_unquantized = torch.nn.parallel.DataParallel(alexnet_unquantized)


#Train a wide-resNet with quantized distillation
quant_distilled_model_name = 'alexnet_quant_distilled{}bits'.format(NUM_BITS)
quantDistilledModelPath = os.path.join(imageNet12modelsFolder, quant_distilled_model_name)
quantDistilledOptions = {}
quant_distilled_model = torchvision.models.alexnet(pretrained=False)

if USE_CUDA:
    quant_distilled_model = quant_distilled_model.cuda()
if NUM_GPUS > 1:
    quant_distilled_model = torch.nn.parallel.DataParallel(quant_distilled_model)

if not quant_distilled_model_name in imagenet_manager.saved_models:
    imagenet_manager.add_new_model(quant_distilled_model_name, quantDistilledModelPath,
                                   arguments_creator_function=quantDistilledOptions)

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
                                                       'teacher_model': alexnet_unquantized,
                                                       'quantizeWeights':True,
                                                       'numBits':NUM_BITS,
                                                       'bucket_size':256,
                                                       'quantize_first_and_last_layer': False},
                             train_loader=train_loader, test_loader=test_loader)
