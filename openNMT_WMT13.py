import os
import torch
import datasets
import translation_models.model as tmm
import translation_models.help_fun as transl_hf
import onmt
import model_manager
import quantization
import copy
import functools
import quantization.help_functions as qhf
import helpers.functions as mhf

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'

USE_CUDA = torch.cuda.is_available()
NUM_GPUS = len(cuda_devices)

TRAIN_TEACHER_MODEL = False
TRAIN_SMALLER_MODEL = False
TRAIN_SEQUENCE_DISTILLED = False
TRAIN_WORD_DISTILLED = False
TRAIN_QUANTIZED_DISTILLED = False
TRAIN_DIFFERENTIABLE_QUANTIZATION = False
COMPUTE_BLEU_MODELS = True
CHECK_PM_QUANTIZATION = True

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

epochsToTrainOnmtIntegDataset = 15
onmtManager = model_manager.ModelManager('model_manager_WMT13.tst',
                                         'model_manager', create_new_model_manager=False)
for x in onmtManager.list_models():
    if onmtManager.get_num_training_runs(x) > 0:
        print(x, onmtManager.load_metadata(x)[1]['perplexity'][-1])

WMT13_saved_models_folder = os.path.join(SAVED_MODELS_FOLDER, 'WMT13')
try:
    os.mkdir(WMT13_saved_models_folder)
except:pass

#load the data
batch_size = 64 * NUM_GPUS
if batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))

transl_dataset = datasets.WMT13_DE_EN(pin_memory=True)
train_loader, test_loader = transl_dataset.getTrainLoader(batch_size), transl_dataset.getTestLoader(batch_size)

#Teacher model
teacherOptions = copy.deepcopy(onmt.standard_options.stdOptions)
#it only matter in the creation of the distillation dataset
teacherOptions['rnn_size'] = 500
teacherOptions['epochs'] = epochsToTrainOnmtIntegDataset
teacherModel_name = 'WMT13_teacherModel'
teacherModelPath = os.path.join(WMT13_saved_models_folder, teacherModel_name)
teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
if USE_CUDA: teacherModel = teacherModel.cuda()
if teacherModel_name not in onmtManager.saved_models:
    onmtManager.add_new_model(teacherModel_name, teacherModelPath,
                                 arguments_creator_function=teacherOptions)
if TRAIN_TEACHER_MODEL:
    onmtManager.train_model(teacherModel, model_name=teacherModel_name,
                               train_function=tmm.train_model,
                               arguments_train_function={'options':teacherOptions},
                               train_loader=train_loader, test_loader=test_loader)
if onmtManager.get_num_training_runs(teacherModel_name) > 0:
    teacherModel.load_state_dict(onmtManager.load_model_state_dict(teacherModel_name))

standardTranslateOptions = onmt.standard_options.standardTranslationOptions


# Smaller model with 1 LSTM layers (1 encoder, 1 for decoder, so in total 2)
# with 500 rnn size (just like the teacher)
#smaller model
smallerOptions = copy.deepcopy(onmt.standard_options.stdOptions)
#if not specified, it was trained with 2 layers (2 for encoder and 2 for decoder, that is) with rnn size of 200
smallerOptions['batch_size'] = batch_size
smallerOptions['rnn_size'] = 500
smallerOptions['layers'] = 1
smallerOptions['epochs'] = 5
smaller_model_name = 'WMT13_smallerModel_{}rnn_size1_layer_5epochs'.format(500)
smallerModelPath = os.path.join(WMT13_saved_models_folder, smaller_model_name)
smallerModel = tmm.create_model(transl_dataset.fields, options=smallerOptions)
if USE_CUDA: smallerModel = smallerModel.cuda()
if smaller_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(smaller_model_name, smallerModelPath,
                              arguments_creator_function=smallerOptions)
if TRAIN_SMALLER_MODEL:
    onmtManager.train_model(smallerModel, model_name=smaller_model_name,
                            train_function=tmm.train_model,
                            arguments_train_function={'options':smallerOptions},
                            train_loader=train_loader, test_loader=test_loader)
if onmtManager.get_num_training_runs(smaller_model_name) > 0:
    smallerModel.load_state_dict(onmtManager.load_model_state_dict(smaller_model_name))
del smallerModel

#Just distilled
distilledOptions = copy.deepcopy(smallerOptions)
distilledOptions['rnn_size'] = 550
distilledOptions['layers'] = 1
distilledOptions['epochs'] = 5
distilled_model_name = 'WMT13_distilledModel_word_level_{}rnn_size1_layer'.format(550)
distilled_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_model_word_level = distilled_model_word_level.cuda()
distilledModelPath = os.path.join(WMT13_saved_models_folder, distilled_model_name)
if distilled_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
if TRAIN_WORD_DISTILLED:
    onmtManager.train_model(distilled_model_word_level, model_name=distilled_model_name,
                               train_function=tmm.train_model,
                               arguments_train_function={'options':distilledOptions,
                                                         'teacher_model': teacherModel,
                                                         'use_distillation_loss':True},
                               train_loader=train_loader, test_loader=test_loader)

if onmtManager.get_num_training_runs(distilled_model_name) > 0:
    distilled_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
del distilled_model_word_level

# For the distilled quantized model we increase the rnn size; sort of like increasing filters
distilledOptions = copy.deepcopy(smallerOptions)
distilledOptions['rnn_size'] = 550
distilledOptions['epochs'] = 5
distilled_model_name_quantized = 'WMT13_distilledModel_word_level_quantized{}bits{}rnn_size1_layer'.format(
                                                                                2, 550)
distilled_quantized_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_quantized_model_word_level = distilled_quantized_model_word_level.cuda()
distilledModelPath = os.path.join(WMT13_saved_models_folder, distilled_model_name_quantized)
if distilled_model_name_quantized not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name_quantized, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
if TRAIN_QUANTIZED_DISTILLED:
    onmtManager.train_model(distilled_quantized_model_word_level, model_name=distilled_model_name_quantized,
                            train_function=tmm.train_model,
                            arguments_train_function={'options':distilledOptions,
                                                         'teacher_model': teacherModel,
                                                         'use_distillation_loss':True,
                                                         'quantizeWeights':True,
                                                         'numBits': 2,
                                                         'bucket_size':256,
                                                         'quantize_first_and_last_layer':False},
                            train_loader=train_loader, test_loader=test_loader)

if onmtManager.get_num_training_runs(distilled_model_name_quantized) > 0:
    distilled_quantized_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name_quantized))
del distilled_quantized_model_word_level

#print bleu for the models
example_translations=False
file_results = 'results_file_BLEU_models_WMT13'
if COMPUTE_BLEU_MODELS:
    with open(file_results, 'a') as fr:
        fr.write('\n\n== New Testing Run == 29 Dec 2017 == \n\n')

for x in onmtManager.list_models():
    if onmtManager.get_num_training_runs(x) == 0:
        continue
    modelOptions = onmtManager.load_metadata(x, 0)[0]
    for key, val in modelOptions.items(): #remeding to an old bug in save_metadata function
        if val == 'None':
            modelOptions[key] = None

    dataset = transl_dataset

    model = tmm.create_model(dataset.fields, options=modelOptions)
    if USE_CUDA: model = model.cuda()
    model.load_state_dict(onmtManager.load_model_state_dict(x, 1))
    if example_translations:
        print('Example of translation for model: "{}"'.format(x))
        num_examples = 5
        linesToTranslate, translated_lines, referenceLines = transl_hf.get_translation_examples(model,
                                                                                                dataset,
                                                                                                num_examples,
                                                                                                modelOptions,
                                                                                                standardTranslateOptions,
                                                                                                shuffle_examples=False)
        print('Original Sentences == Translation == Ref Translation')
        print('\n'.join(' == '.join(x) for x in zip(linesToTranslate, translated_lines, referenceLines)))
    if COMPUTE_BLEU_MODELS:
        bleu = transl_hf.get_bleu_model(model, dataset, modelOptions, standardTranslateOptions)
    else:
        bleu = 'Not computed'

    perplexity = onmtManager.load_metadata(x,1)[1]['perplexity'][-1]
    str_to_save = 'Model "{}"  ==> Perplexity: {}, BLEU: {}'.format(x, perplexity, bleu)
    if COMPUTE_BLEU_MODELS:
        with open(file_results, 'a') as fr:
            fr.write(str_to_save + '\n')
    print(str_to_save)

    curr_num_bit = onmtManager.load_metadata(x)[0].get('numBits', None)
    if curr_num_bit is not None:
        quant_fun = functools.partial(quantization.uniformQuantization, s=2**curr_num_bit, bucket_size=256)
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                       'uniform', s=2**curr_num_bit)
        print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                            mhf.get_size_reduction(actual_bit_huffmman, bucket_size=256)))
        print('Size MB: {}'.format(mhf.get_size_quantized_model(model, curr_num_bit, quant_fun, 256, quantizeFirstLastLayer=False)))

    if CHECK_PM_QUANTIZATION:
        QUANTIZE_FIRST_LAST_LAYER = True
        if 'distilledModel_word_level' in x:
            for numBit in [2]:
                for bucket_size in (None, 256):
                    model.load_state_dict(onmtManager.load_model_state_dict(x, 1))
                    numParam = sum(1 for _ in model.parameters())
                    for idx, p in enumerate(model.parameters()):
                        if QUANTIZE_FIRST_LAST_LAYER is False:
                            if idx == 0 or idx == numParam - 1:
                                continue
                        p.data = quantization.uniformQuantization(p.data, s=2**numBit, type_of_scaling='linear',
                                                                  bucket_size=bucket_size)[0]
                    perplexity = tmm.evaluate_model(model, test_loader).ppl()
                    if COMPUTE_BLEU_MODELS:
                        bleu = transl_hf.get_bleu_model(model, dataset, modelOptions, standardTranslateOptions)
                    else:
                        bleu = 'Not Computed'
                    str_to_save = 'PM quantization of model "{}" with "{}" bits and bucket size {}: Perplexity : {}, BLEU: {}'.format(
                        x, numBit, bucket_size, perplexity, bleu)
                    quant_fun = functools.partial(quantization.uniformQuantization, s=2**numBit, bucket_size=bucket_size)
                    actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                                   'uniform', s=2**numBit)
                    size_reduction = mhf.get_size_reduction(actual_bit_huffmman, bucket_size=bucket_size)
                    size_mb = mhf.get_size_quantized_model(model, numBit, quant_fun, bucket_size, quantizeFirstLastLayer=QUANTIZE_FIRST_LAST_LAYER)
                    str_to_save += '\n' + 'Effective bit Huffman: {} - Size reduction: {} - Size MB: {}'.format(actual_bit_huffmman,size_reduction, size_mb)
                    if COMPUTE_BLEU_MODELS:
                        with open(file_results, 'a') as fr:
                            fr.write(str_to_save + '\n')
                    print(str_to_save)

#now for the models trained with the differentiable quantization algorithm
# list_distilled_models = ['WMT13_distilledModel_word_level_{}rnn_size1_layer'.format(x)
#                          for x in rnn_sizes]
# optQuanPointOptions = copy.deepcopy(onmt.onmt.standard_options.stdOptions)
# for idx_model_distilled, distilled_model_name_to_quantize in enumerate(list_distilled_models):
#     modelOptions = onmtManager.load_metadata(distilled_model_name_to_quantize, 0)[0]
#     for key, val in modelOptions.items():  # remeding to an old bug in save_metadata function
#         if val == 'None':
#             modelOptions[key] = None
#     dataset = transl_dataset #since we don't use sequence level distillation
#     for numBit in numBits:
#         if numBit == 8: continue
#         save_path = onmtManager.get_model_base_path(distilled_model_name_to_quantize) + \
#                     'quant_points_{}bit_bucket_size256'.format(numBit)
#         with open(save_path, 'rb') as p:
#             quantization_points, infoDict = pickle.load(p)
#         distilledModel = tmm.create_model(dataset.fields, options=modelOptions)
#         distilledModel.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name_to_quantize))
#         if USE_CUDA: distilledModel = distilledModel.cuda()
#         for idx, p in enumerate(distilledModel.parameters()):
#             p.data = quantization.nonUniformQuantization(p.data, quantization_points[idx], bucket_size=256)[0]
#         reported_perplexity = infoDict['perplexity'][-1]
#         perplexity = tmm.evaluate_model(distilledModel, test_loader).ppl()
#         if COMPUTE_BLEU_MODELS:
#             bleu = transl_hf.get_bleu_model(distilledModel, dataset, optQuanPointOptions, standardTranslateOptions)
#         else:
#             bleu = 'Not Computed'
#         str_to_save = 'Model "{}"  ==> Reported perplexity : {}, Actual perplexity: {}, BLEU: {}'.format(
#             distilled_model_name_to_quantize + 'quant_points_{}bit_bucket_size256'.format(numBit),
#             reported_perplexity, perplexity, bleu)
#         if COMPUTE_BLEU_MODELS:
#             with open(file_results, 'a') as fr:
#                 fr.write(str_to_save + '\n')
#         print(str_to_save)
#
#         quantization_functions = [functools.partial(quantization.nonUniformQuantization,
#                                                     listQuantizationPoints=qp,
#                                                     bucket_size=256) for qp in quantization_points]
#         actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(distilledModel.parameters(),
#                                                                        quantization_functions,
#                                                                        'nonUniform')
#         print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
#                                                                       mhf.get_size_reduction(
#                                                                           actual_bit_huffmman,
#                                                                           bucket_size=256)))
