import model_manager
import torch
import os
import datasets
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
from cnn_models.wide_resnet import Wide_ResNet
import quantization
import pickle
import copy
import functools
import quantization.help_functions as qhf
import helpers.functions as mhf

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'
USE_CUDA = torch.cuda.is_available()

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))
NUM_GPUS = len(cuda_devices)

CHECK_PM_QUANTIZATION = True

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

cifar100Manager = model_manager.ModelManager('model_manager_cifar100.tst',
                                            'model_manager', create_new_model_manager=False)
cifar100modelsFolder = os.path.join(SAVED_MODELS_FOLDER, 'cifar100')

for x in cifar100Manager.list_models():
    if cifar100Manager.get_num_training_runs(x) >= 1:
        s = '{}; Last prediction acc: {}, Best prediction acc: {}'.format(x,
                                            cifar100Manager.load_metadata(x)[1]['predictionAccuracy'][-1],
                                            max(cifar100Manager.load_metadata(x)[1]['predictionAccuracy']))
        print(s)

try:
    os.mkdir(cifar100modelsFolder)
except:pass

epochsToTrainCIFAR100 = 200
epochsToTrainCIFAR100_diffquant = 20

batch_size = 110
if batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))

cifar100 = datasets.CIFAR100()
train_loader, test_loader = cifar100.getTrainLoader(batch_size), cifar100.getTestLoader(batch_size)

TRAIN_TEACHER_MODEL = False
TRAIN_DISTILLED_MODEL = True

# Teacher model
teacher_model_name = 'cifar100_teacher_new'
teacherModelPath = os.path.join(cifar100modelsFolder, teacher_model_name)
teacherOptions = {'widen_factor':20, 'depth':28, 'dropout_rate':0.3, 'num_classes':100}
teacherModel = Wide_ResNet(**teacherOptions)
if USE_CUDA: teacherModel = teacherModel.cuda()
if NUM_GPUS > 1:
    teacherModel = torch.nn.parallel.DataParallel(teacherModel)
if not teacher_model_name in cifar100Manager.saved_models:
    cifar100Manager.add_new_model(teacher_model_name, teacherModelPath,
            arguments_creator_function=teacherOptions)
if TRAIN_TEACHER_MODEL:
    cifar100Manager.train_model(teacherModel, model_name=teacher_model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrainCIFAR100,
                                                         'initial_learning_rate':0.1,
                                                         'print_every':50,
                                                         'learning_rate_style':'cifar100',
                                                         'weight_decayL2': 0.0005,
                                                         'start_epoch':0},
                               train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(cifar100Manager.load_model_state_dict(teacher_model_name))


specSmallerModels0 = {'widen_factor':8, 'depth':22, 'dropout_rate':0.3, 'num_classes':100}
specSmallerModels1 = {'widen_factor':6, 'depth':16, 'dropout_rate':0.3, 'num_classes':100}
specSmallerModels2 = {'widen_factor':4, 'depth':10, 'dropout_rate':0.3, 'num_classes':100}
specSmallerModels3 = {'widen_factor':10, 'depth':16, 'dropout_rate':0.3, 'num_classes':100}

specSmallerModels = [specSmallerModels0, specSmallerModels1, specSmallerModels2, specSmallerModels3]

numBits = [4]
for idx_spec, model_spec in enumerate(specSmallerModels):
    if idx_spec <= 2: continue
    #smaller model
    # model_name = 'cifar100_smaller_spec{}'.format(idx_spec)
    # smallerModelPath = os.path.join(cifar100modelsFolder, model_name)
    # smallerModel = Wide_ResNet(**model_spec)
    # if USE_CUDA: smallerModel = smallerModel.cuda()
    # if not model_name in cifar100Manager.saved_models:
    #     cifar100Manager.add_new_model(model_name, smallerModelPath,
    #             arguments_creator_function=model_spec)
    # cifar100Manager.train_model(smallerModel, model_name=model_name,
    #                            train_function=convForwModel.train_model,
    #                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR100,
    #                                                      'initial_learning_rate':0.1,
    #                                                      'print_every':100,
    #                                                      'learning_rate_style':'cifar100',
    #                                                      'weight_decayL2': 0.0005},
    #                            train_loader=train_loader, test_loader=test_loader)
    # smallerModel.load_state_dict(cifar100Manager.load_model_state_dict(model_name))

    #distilled model
    distilled_model_name = 'cifar100_distilled_spec{}'.format(idx_spec)
    # distilledModelPath = os.path.join(cifar100modelsFolder, distilled_model_name)
    # distilledModel = Wide_ResNet(**model_spec)
    # if USE_CUDA: distilledModel = distilledModel.cuda()
    # if not distilled_model_name in cifar100Manager.saved_models:
    #     cifar100Manager.add_new_model(distilled_model_name, distilledModelPath,
    #             arguments_creator_function=model_spec)
    # cifar100Manager.train_model(distilledModel, model_name=distilled_model_name,
    #                            train_function=convForwModel.train_model,
    #                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR100,
    #                                                      'initial_learning_rate':0.1,
    #                                                      'print_every':50,
    #                                                      'learning_rate_style':'cifar100',
    #                                                      'weight_decayL2': 0.0005,
    #                                                      'use_distillation_loss':True,
    #                                                      'teacher_model':teacherModel,
    #                                                      },
    #                            train_loader=train_loader, test_loader=test_loader)
    # distilledModel.load_state_dict(cifar100Manager.load_model_state_dict(distilled_model_name))


    for numBit in numBits:
        #quantized distillation
        distilled_quantized_model_name = distilled_model_name + '_quant{}bits_new'.format(numBit)
        distilled_quantizedModelPath = os.path.join(cifar100modelsFolder, distilled_quantized_model_name)
        distilled_quantizedModel = Wide_ResNet(**model_spec)
        if USE_CUDA: distilled_quantizedModel = distilled_quantizedModel.cuda()
        if NUM_GPUS > 1: distilled_quantizedModel = torch.nn.parallel.DataParallel(distilled_quantizedModel)
        if not distilled_quantized_model_name in cifar100Manager.saved_models:
            cifar100Manager.add_new_model(distilled_quantized_model_name, distilled_quantizedModelPath,
                                          arguments_creator_function=model_spec)
        if TRAIN_DISTILLED_MODEL:
            cifar100Manager.train_model(distilled_quantizedModel, model_name=distilled_quantized_model_name,
                                        train_function=convForwModel.train_model,
                                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR100,
                                                                  'initial_learning_rate': 0.1,
                                                                  'print_every': 50,
                                                                  'learning_rate_style': 'cifar100',
                                                                  'weight_decayL2': 0.0005,
                                                                  'use_distillation_loss': True,
                                                                  'teacher_model': teacherModel,
                                                                  'quantizeWeights': True,
                                                                  'numBits': numBit,
                                                                  'bucket_size': 256,
                                                                  'quantize_first_and_last_layer': False,
                                                                  'start_epoch': 0},
                                        train_loader=train_loader, test_loader=test_loader)
        distilled_quantizedModel.load_state_dict(cifar100Manager.load_model_state_dict(distilled_quantized_model_name))

        #differentiable quantization
        # numPointsPerTensor = 2 ** numBit
        # distilled_quantizedModel = Wide_ResNet(**model_spec)
        # if USE_CUDA: distilled_quantizedModel = distilled_quantizedModel.cuda()
        # distilled_quantizedModel.load_state_dict(cifar100Manager.load_model_state_dict(distilled_model_name))
        # quantized_model_dict, quantization_points, infoDict = convForwModel.optimize_quantization_points(
        #                                                               distilled_quantizedModel,
        #                                                               train_loader, test_loader,
        #                                                               numPointsPerTensor=numPointsPerTensor,
        #                                                               assignBitsAutomatically=True,
        #                                                               bucket_size=256,
        #                                                               epochs_to_train=epochsToTrainCIFAR100_diffquant,
        #                                                               initial_learning_rate=1e-4,
        #                                                               print_every=100,
        #                                                               use_distillation_loss=True,
        #                                                               learning_rate_style='quant_points_cifar100')
        # quantization_points = [x.data.view(1,-1).cpu().numpy().tolist()[0] for x in quantization_points]
        # save_path = cifar100Manager.get_model_base_path(distilled_model_name) + \
        #                         '_diffquant_points_{}bits'.format(numBit)
        # with open(save_path, 'wb') as p:
        #     pickle.dump((quantization_points, infoDict), p)
        # torch.save(quantized_model_dict, save_path+'_model_state_dict')

raise ValueError

def load_model_from_name(x):
    opt = cifar100Manager.load_metadata(x, 0)[0]
    #small old bug in the saving of metadata, this is a cheap trick to remedy it
    for key, val in opt.items():
        if isinstance(val, str):
            opt[key] = eval(val)
    model = Wide_ResNet(**opt)
    if USE_CUDA: model = model.cuda()
    model.load_state_dict(cifar100Manager.load_model_state_dict(x))
    return model

for x in cifar100Manager.list_models():
    if cifar100Manager.get_num_training_runs(x) == 0:
        continue
    model = load_model_from_name(x)
    reported_accuracy = cifar100Manager.load_metadata(x)[1]['predictionAccuracy'][-1]
    pred_accuracy = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
    print('Model "{}" ==> Prediction accuracy: {:2f}% == Reported accuracy: {:2f}%'.format(x,
                                                        pred_accuracy*100, reported_accuracy*100))
    curr_num_bit = cifar100Manager.load_metadata(x)[0].get('numBits', None)
    if curr_num_bit is not None:
        quant_fun = functools.partial(quantization.uniformQuantization, s=2**curr_num_bit, bucket_size=256)
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                       'uniform', s=2**curr_num_bit)
        print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                            mhf.get_size_reduction(actual_bit_huffmman, bucket_size=256)))
    if CHECK_PM_QUANTIZATION:
        if 'distilled' in x and 'quant' not in x:
            for numBit in numBits:
                for bucket_size in [None, 256]:
                    model.load_state_dict(cifar100Manager.load_model_state_dict(x))
                    for p in model.parameters():
                        p.data = quantization.uniformQuantization(p.data, s=2**numBit, type_of_scaling='linear',
                                                                  bucket_size=bucket_size)[0]
                    predAcc = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
                    print('PM quantization of model "{}" with "{}" bits and bucket {}: {:2f}%'.format(x,
                                                                                                      numBit,
                                                                                                      bucket_size,
                                                                                                      predAcc * 100))
                    quant_fun = functools.partial(quantization.uniformQuantization, s=2**numBit, bucket_size=bucket_size)
                    actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                                   'uniform',s=2**numBit)
                    print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                                                                  mhf.get_size_reduction(
                                                                                      actual_bit_huffmman,
                                                                                      bucket_size=bucket_size)))


#check diff quantization
distilled_model_names = ['cifar100_distilled_spec{}'.format(idx_spec) for idx_spec in range(len(specSmallerModels))]
for distilled_model_name in distilled_model_names:
    modelOptions = cifar100Manager.load_metadata(distilled_model_name, 0)[0]
    # small old bug in the saving of metadata, this is a cheap trick to remedy it
    for key, val in modelOptions.items():
        if isinstance(val, str):
            modelOptions[key] = eval(val)
    for numBit in numBits:
        if numBit == 8: continue
        distilled_quantized_model = Wide_ResNet(**modelOptions)
        if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
        save_path = cifar100Manager.get_model_base_path(distilled_model_name) + \
                                '_diffquant_points_{}bits'.format(numBit)
        with open(save_path, 'rb') as p:
            quantization_points, infoDict = pickle.load(p)
        distilled_quantized_model.load_state_dict(torch.load(save_path + '_model_state_dict'))

        quantization_functions = [functools.partial(quantization.nonUniformQuantization,
                                                    listQuantizationPoints=qp,
                                                    bucket_size=256) for qp in quantization_points]
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(distilled_quantized_model.parameters(),
                                                                     quantization_functions,
                                                                     'nonUniform')
        pred_accuracy = cnn_hf.evaluateModel(distilled_quantized_model, test_loader, fastEvaluation=False)
        print('Differentiable Quantization of model "{}" with {} bits ==> Prediction accuracy: {:2f}% '.format(
                                                            distilled_model_name,
                                                            numBit,
                                                            pred_accuracy * 100))
        print('Effective bit huffman: {}'.format(actual_bit_huffmman))
