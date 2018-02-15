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

epochsToTrainCIFAR = 200
USE_BATCH_NORM = True
AFFINE_BATCH_NORM = True

TRAIN_TEACHER_MODEL = False
TRAIN_SMALLER_MODEL = False
TRAIN_SMALLER_QUANTIZED_MODEL = False
TRAIN_DISTILLED_MODEL = False
TRAIN_DIFFERENTIABLE_QUANTIZATION = False
CHECK_PM_QUANTIZATION = True

batch_size = 25
cifar10 = datasets.CIFAR10()
train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)

# Teacher model
model_name = 'cifar10_teacher'
teacherModelPath = os.path.join(cifar10modelsFolder, model_name)
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=USE_BATCH_NORM,
                                              useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
if USE_CUDA: teacherModel = teacherModel.cuda()
if not model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(model_name, teacherModelPath,
            arguments_creator_function={**convForwModel.teacherModelSpec,
                                        'useBatchNorm':USE_BATCH_NORM,
                                        'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
if TRAIN_TEACHER_MODEL:
    cifar10Manager.train_model(teacherModel, model_name=model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
                               train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(cifar10Manager.load_model_state_dict(model_name))
cnn_hf.evaluateModel(teacherModel, test_loader, k=5)


#Define the architechtures we want to try
smallerModelSpec0 = {'spec_conv_layers': [(75, 5, 5), (50, 5, 5), (50, 5, 5), (25, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [500], 'width': 32, 'height': 32}
smallerModelSpec1 = {'spec_conv_layers': [(50, 5, 5), (25, 5, 5), (25, 5, 5), (10, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [400], 'width': 32, 'height': 32}
smallerModelSpec2 = {'spec_conv_layers': [(25, 5, 5), (10, 5, 5), (10, 5, 5), (5, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [300], 'width': 32, 'height': 32}

smallerModelSpecs = [smallerModelSpec0, smallerModelSpec1, smallerModelSpec2]

# distilled + quantized better than quantized:
# for numBit in numBits:
#     if numBit == 8:continue
#     model_name = 'cifar10_smaller_spec0_quantized{}bitsNoBucketing'.format(numBit)
#     quantized_model_path = os.path.join(cifar10modelsFolder, model_name)
#     quantized_model = convForwModel.ConvolForwardNet(**smallerModelSpec0,
#                                                   useBatchNorm=USE_BATCH_NORM,
#                                                   useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#     if USE_CUDA: quantized_model = quantized_model.cuda()
#     if not model_name in cifar10Manager.saved_models:
#         cifar10Manager.add_new_model(model_name, quantized_model_path,
#                 arguments_creator_function={**smallerModelSpec0,
#                                             'useBatchNorm':USE_BATCH_NORM,
#                                             'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
#     if TRAIN_SMALLER_QUANTIZED_MODEL:
#         for _ in range(2):
#             cifar10Manager.train_model(quantized_model, model_name=model_name,
#                                        train_function=convForwModel.train_model,
#                                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
#                                                                  'quantizeWeights':True,
#                                                                  'numBits':numBit,
#                                                                  'bucket_size':None},
#                                        train_loader=train_loader, test_loader=test_loader)
#     quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(model_name))


# Test with deeper student
deeper_student_spec = {'spec_conv_layers': [(76, 3, 3), (76, 3, 3),(76, 3, 3), (126, 3, 3), (126, 3, 3), (126, 3, 3),
                                             (148, 3, 3), (148, 3, 3), (148, 3, 3), (148, 3, 3), (148, 3, 3)],
                    'spec_max_pooling': [(2,2,2), (5, 2, 2), (10, 2, 2)],
                    'spec_dropout_rates': [(2, 0.2), (5, 0.3), (10, 0.35), (11, 0.4), (12, 0.4)],
                    'spec_linear': [1000, 1000, 1000], 'width': 32, 'height': 32}

numBits = [4, 2]
#train normal distilled
# model = convForwModel.ConvolForwardNet(**deeper_student_spec,
#                                        useBatchNorm=USE_BATCH_NORM,
#                                        useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
# model_name = 'cifar10_deeper_normal'
# model_path = os.path.join(cifar10modelsFolder, model_name)
# if USE_CUDA: model = model.cuda()
# if not model_name in cifar10Manager.saved_models:
#         cifar10Manager.add_new_model(model_name, model_path,
#                 arguments_creator_function={**deeper_student_spec,
#                                             'useBatchNorm':USE_BATCH_NORM,
#                                             'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
# cifar10Manager.train_model(model, model_name=model_name,
#                            train_function=convForwModel.train_model,
#                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
#                            train_loader=train_loader, test_loader=test_loader)
# raise ValueError

# for numBit in numBits:
#     model = convForwModel.ConvolForwardNet(**deeper_student_spec,
#                                            useBatchNorm=USE_BATCH_NORM,
#                                            useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#     model_name = 'cifar10_deeper_distilled_quantized{}bits'.format(numBit)
#     model_path = os.path.join(cifar10modelsFolder, model_name)
#     if USE_CUDA: model = model.cuda()
#     if not model_name in cifar10Manager.saved_models:
#             cifar10Manager.add_new_model(model_name, model_path,
#                     arguments_creator_function={**deeper_student_spec,
#                                                 'useBatchNorm':USE_BATCH_NORM,
#                                                 'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
#     cifar10Manager.train_model(model, model_name=model_name,
#                                train_function=convForwModel.train_model,
#                                arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
#                                                          'quantizeWeights': True,
#                                                          'numBits': numBit,
#                                                          'bucket_size': 256,
#                                                          'use_distillation_loss':True,
#                                                          'teacher_model':teacherModel,
#                                                          'quantize_first_and_last_layer':False},
#                                train_loader=train_loader, test_loader=test_loader)
#     print('End simple test')


# for idx_spec, model_spec in enumerate(smallerModelSpecs):
#
#     model_name = 'cifar10_smaller_spec{}'.format(idx_spec)
#
#     smallerModelPath = os.path.join(cifar10modelsFolder, model_name)
#     smallerModel = convForwModel.ConvolForwardNet(**model_spec,
#                                                   useBatchNorm=USE_BATCH_NORM,
#                                                   useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#     if USE_CUDA: smallerModel = smallerModel.cuda()
#     if not model_name in cifar10Manager.saved_models:
#         cifar10Manager.add_new_model(model_name, smallerModelPath,
#                 arguments_creator_function={**model_spec,
#                                             'useBatchNorm':USE_BATCH_NORM,
#                                             'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
#     if TRAIN_SMALLER_MODEL:
#         for _ in range(2):
#             cifar10Manager.train_model(smallerModel, model_name=model_name,
#                                        train_function=convForwModel.train_model,
#                                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
#                                        train_loader=train_loader, test_loader=test_loader)
#     smallerModel.load_state_dict(cifar10Manager.load_model_state_dict(model_name))
#
#     #free up some memory
#     del smallerModel
#
#     distilledModelSpec = copy.deepcopy(model_spec)
#     distilledModelSpec['spec_dropout_rates'] = [] #no dropout with distilled model
#
#     ## distilled model
#     distilled_model_name = 'cifar10_distilled_spec{}'.format(idx_spec)
#
#     distilledModelPath = os.path.join(cifar10modelsFolder, distilled_model_name)
#     distilledModel = convForwModel.ConvolForwardNet(**distilledModelSpec,
#                                                     useBatchNorm=USE_BATCH_NORM,
#                                                     useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#     if USE_CUDA: distilledModel = distilledModel.cuda()
#     if not distilled_model_name in cifar10Manager.saved_models:
#         cifar10Manager.add_new_model(distilled_model_name, distilledModelPath,
#                 arguments_creator_function={**distilledModelSpec,
#                                             'useBatchNorm':USE_BATCH_NORM,
#                                             'useAffineTransformInBatchNorm':AFFINE_BATCH_NORM})
#     if TRAIN_DISTILLED_MODEL:
#         for _ in range(2):
#             cifar10Manager.train_model(distilledModel, model_name=distilled_model_name,
#                                        train_function=convForwModel.train_model,
#                                        arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
#                                                                  'teacher_model':teacherModel,
#                                                                  'use_distillation_loss':True},
#                                        train_loader=train_loader, test_loader=test_loader)
#     distilledModel.load_state_dict(cifar10Manager.load_model_state_dict(distilled_model_name))
#
#     for numBit in numBits:
#         distilled_quantized_model_name = 'cifar10_distilled_spec{}_quantized{}bits'.format(idx_spec, numBit)
#
#         distilled_quantized_model_path = os.path.join(cifar10modelsFolder, distilled_quantized_model_name)
#         distilled_quantized_model = convForwModel.ConvolForwardNet(**distilledModelSpec,
#                                                         useBatchNorm=USE_BATCH_NORM,
#                                                         useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#         if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
#         if not distilled_quantized_model_name in cifar10Manager.saved_models:
#             cifar10Manager.add_new_model(distilled_quantized_model_name, distilled_quantized_model_path,
#                                          arguments_creator_function={**distilledModelSpec,
#                                                                      'useBatchNorm': USE_BATCH_NORM,
#                                                                      'useAffineTransformInBatchNorm': AFFINE_BATCH_NORM})
#         if TRAIN_DISTILLED_MODEL:
#             for _ in range(2):
#                 cifar10Manager.train_model(distilled_quantized_model, model_name=distilled_quantized_model_name,
#                                            train_function=convForwModel.train_model,
#                                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
#                                                                      'teacher_model': teacherModel,
#                                                                      'use_distillation_loss': True,
#                                                                      'quantizeWeights':True,
#                                                                      'numBits':numBit,
#                                                                      'bucket_size':256},
#                                            train_loader=train_loader, test_loader=test_loader)
#         distilled_quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(distilled_quantized_model_name))
#         del distilled_quantized_model
#         # optimize quantization points
#         if numBit == 8:  # but no 8 bits with differentiable quantization
#             continue
#
#         if TRAIN_DIFFERENTIABLE_QUANTIZATION:
#             distilled_quantized_model_name = distilled_model_name + '_quant_points_{}bits'.format(numBit)
#             distilled_quantized_model = convForwModel.ConvolForwardNet(**distilledModelSpec,
#                                                                        useBatchNorm=USE_BATCH_NORM,
#                                                                        useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)
#             if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
#             distilled_quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(distilled_model_name))
#             epochs_to_train = 10 if numBit == 4 else 20
#
#             quantized_model_dict, quantization_points, infoDict = convForwModel.optimize_quantization_points(
#                 distilled_quantized_model,
#                 train_loader, test_loader, numPointsPerTensor=2**numBit,
#                 assignBitsAutomatically=True,
#                 bucket_size=256, epochs_to_train=epochs_to_train,
#                 use_distillation_loss=True, initial_learning_rate=1e-6)
#             quantization_points = [x.data.view(1,-1).cpu().numpy().tolist()[0] for x in quantization_points]
#             save_path = cifar10Manager.get_model_base_path(distilled_model_name) + \
#                                     'quant_points_{}bits'.format(numBit)
#             with open(save_path, 'wb') as p:
#                 pickle.dump((quantization_points, infoDict), p)
#             torch.save(quantized_model_dict, save_path+'_model_state_dict')
#
# del teacherModel


#check quality of distilled models.. the accuracy reported is reported with the weights
#at the last iteration that may not have been quantized. Before the weights are returned
#though they are quantized, so there is this little difference.

def load_model_from_name(x):
    opt = cifar10Manager.load_metadata(x, 0)[0]
    #small old bug in the saving of metadata, this is a cheap trick to remedy it
    for key, val in opt.items():
        if isinstance(val, str):
            opt[key] = eval(val)
    model = convForwModel.ConvolForwardNet(**opt)
    if USE_CUDA: model = model.cuda()
    model.load_state_dict(cifar10Manager.load_model_state_dict(x))
    return model

for x in cifar10Manager.list_models():
    if cifar10Manager.get_num_training_runs(x) == 0:
        continue
    model = load_model_from_name(x)
    reported_accuracy = cifar10Manager.load_metadata(x)[1]['predictionAccuracy'][-1]
    pred_accuracy = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
    print('Model "{}" ==> Prediction accuracy: {:2f}% == Reported accuracy: {:2f}%'.format(x,
                                                        pred_accuracy*100, reported_accuracy*100))
    curr_num_bit = cifar10Manager.load_metadata(x)[0].get('numBits', None)
    if curr_num_bit is not None:
        quant_fun = functools.partial(quantization.uniformQuantization, s=2**curr_num_bit, bucket_size=256)
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                       'uniform', s=2**curr_num_bit)
        print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                            mhf.get_size_reduction(actual_bit_huffmman, bucket_size=256)))
    if CHECK_PM_QUANTIZATION:
        QUANTIZE_FIRST_LAST_LAYER = False
        if 'distilled' in x:
            for numBit in numBits:
                for bucket_size in (None, 256):
                    model.load_state_dict(cifar10Manager.load_model_state_dict(x))
                    numParam = sum(1 for _ in model.parameters())
                    for idx, p in enumerate(model.parameters()):
                        if QUANTIZE_FIRST_LAST_LAYER is False:
                            if idx == 0 or idx == numParam - 1:
                                continue
                        p.data = quantization.uniformQuantization(p.data, s=2**numBit, type_of_scaling='linear',
                                                                  bucket_size=bucket_size)[0]
                    predAcc = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
                    print('PM quantization of model "{}" with "{}" bits and {} buckets: {:2f}%'.format(x, numBit,
                                                                                            bucket_size, predAcc * 100))
                    quant_fun = functools.partial(quantization.uniformQuantization, s=2**numBit, bucket_size=bucket_size)
                    actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                                   'uniform',s=2**numBit)
                    size_mb = mhf.get_size_quantized_model(model, numBit, quant_fun, bucket_size,
                                                           quantizeFirstLastLayer=QUANTIZE_FIRST_LAST_LAYER)
                    print('Effective bit Huffman: {} - Size reduction: {} - Size MB: {}'.format(actual_bit_huffmman,
                                                                                  mhf.get_size_reduction(
                                                                                      actual_bit_huffmman,
                                                                                      bucket_size=bucket_size),
                                                                                       size_mb))

distilled_model_names = ['cifar10_distilled_spec{}'.format(idx_spec) for idx_spec in range(len(smallerModelSpecs))]
for distilled_model_name in distilled_model_names:
    modelOptions = cifar10Manager.load_metadata(distilled_model_name, 0)[0]
    # small old bug in the saving of metadata, this is a cheap trick to remedy it
    for key, val in modelOptions.items():
        if isinstance(val, str):
            modelOptions[key] = eval(val)
    for numBit in numBits:
        if numBit == 8: continue
        distilled_quantized_model_name = distilled_model_name + '_quant_points_{}bits'.format(numBit)
        distilled_quantized_model = convForwModel.ConvolForwardNet(**modelOptions)
        if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
        save_path = cifar10Manager.get_model_base_path(distilled_model_name) + \
                    'quant_points_{}bits'.format(numBit)

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
