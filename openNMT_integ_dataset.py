import os
import torch
import datasets
import translation_models.model as tmm
import translation_models.help_fun as transl_hf
import onmt
import model_manager
import quantization
import copy
import pickle
import functools
import quantization.help_functions as qhf
import helpers.functions as mhf

cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))


datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'
USE_CUDA = torch.cuda.is_available()
NUM_GPUS = len(cuda_devices)

TRAIN_TEACHER_MODEL=False
TRAIN_SMALLER_MODEL=False
TRAIN_SEQUENCE_DISTILLED=False
TRAIN_WORD_DISTILLED=False
TRAIN_QUANTIZED_DISTILLED=False
TRAIN_DIFFERENTIABLE_QUANTIZATION=False
CREATE_DISTILLATION_DATASET=False
COMPUTE_BLEU_MODELS = False
CHECK_PM_QUANTIZATION = True
COMPUTE_WORD_PERCENTAGE_SIMILARITY = True

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

epochsToTrainOnmtIntegDataset = 15

onmtManager = model_manager.ModelManager('model_manager_integ_dataset.tst',
                                         'model_manager', create_new_model_manager=False)
for x in onmtManager.list_models():
    if onmtManager.get_num_training_runs(x) > 0:
        print(x, onmtManager.load_metadata(x)[1]['perplexity'][-1])

integ_dataset_saved_models_folder = os.path.join(SAVED_MODELS_FOLDER, 'integ_dataset')
try:
    os.mkdir(integ_dataset_saved_models_folder)
except:pass

#load the data
batch_size = 64 * NUM_GPUS
if batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))

transl_dataset = datasets.onmt_integ_dataset(pin_memory=True)
train_loader, test_loader = transl_dataset.getTrainLoader(batch_size), transl_dataset.getTestLoader(batch_size)

#Teacher model
teacherOptions = copy.deepcopy(onmt.standard_options.stdOptions)
#it only matter in the creation of the distillation dataset
teacherOptions['rnn_size'] = 500
teacherOptions['epochs'] = epochsToTrainOnmtIntegDataset
teacherModel_name = 'integ_dataset_teacherModel'
teacherModelPath = os.path.join(integ_dataset_saved_models_folder, teacherModel_name)
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

#now create a distillation dataset
standardTranslateOptions = onmt.standard_options.standardTranslationOptions

create_distilled_dataset_options = copy.deepcopy(teacherOptions)
folder_distillation_dataset = os.path.join(transl_dataset.dataFolder, 'distilled_dataset_' + teacherModel_name)
if CREATE_DISTILLATION_DATASET:
    print('Creating distillation dataset from scratch')
    transl_hf.create_distillation_dataset(teacherModel, create_distilled_dataset_options, standardTranslateOptions,
                                            transl_dataset, folder_distillation_dataset)
    print('Distillation dataset created')

try:
    distilled_dataset = datasets.translation_datasets.TranslationDataset(folder_distillation_dataset, src_language='de',
                                                                     tgt_language='en', pin_memory=True)
    train_distilled_loader, test_distilled_loader = distilled_dataset.getTrainLoader(batch_size), distilled_dataset.getTestLoader(batch_size)
    print('Distillation dataset loaded')
except:
    print('Problems loading the distillation dataset')
    #just so they don't raise errors..
    distilled_dataset = transl_dataset
    train_distilled_loader = train_loader
    test_distilled_loader = test_loader

# quick last minute experiment of distill vs normal loss
# smallerOptions = copy.deepcopy(onmt.standard_options.stdOptions)
# #if not specified, it was trained with 2 layers (2 for encoder and 2 for decoder, that is) with rnn size of 200
# smallerOptions['batch_size'] = batch_size
# smallerOptions['rnn_size'] = 512
# smallerOptions['layers'] = 1
# smallerOptions['epochs'] = epochsToTrainOnmtIntegDataset
# for numBit in [4]:
#     model_name = 'integ_dataset_smallerModel_{}rnn_size1_layer_quantized{}bits'.format(512, numBit)
#     smallerModelPath = os.path.join(integ_dataset_saved_models_folder, model_name)
#     smallerModel = tmm.create_model(transl_dataset.fields, options=smallerOptions)
#     if USE_CUDA: smallerModel = smallerModel.cuda()
#     if model_name not in onmtManager.saved_models:
#         onmtManager.add_new_model(model_name, smallerModelPath,
#                                      arguments_creator_function=smallerOptions)
#     onmtManager.train_model(smallerModel, model_name=model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':smallerOptions,
#                                                          'quantizeWeights': True,
#                                                          'numBits':numBit,
#                                                          'bucket_size':256},
#                                train_loader=train_loader, test_loader=test_loader)
#     if onmtManager.get_num_training_runs(model_name) > 0:
#         smallerModel.load_state_dict(onmtManager.load_model_state_dict(model_name))
#     print('finished training, computing BLEU')
#     bleu = transl_hf.get_bleu_model(smallerModel, transl_dataset, smallerOptions, standardTranslateOptions)
#     bleu='not computed'
#     ppl = tmm.evaluate_model(smallerModel, test_loader).ppl()
#     print('BLEU is : {}'.format(bleu))


del teacherModel


rnn_sizes = [128, 256, 512]
numBits = [2,4,8]
# for rnn_size in rnn_sizes:
#     #smaller model
#     smallerOptions = copy.deepcopy(onmt.standard_options.stdOptions)
#     #if not specified, it was trained with 2 layers (2 for encoder and 2 for decoder, that is) with rnn size of 200
#     smallerOptions['batch_size'] = batch_size
#     smallerOptions['rnn_size'] = rnn_size
#     smallerOptions['layers'] = 1
#     smallerOptions['epochs'] = epochsToTrainOnmtIntegDataset
#     model_name = 'integ_dataset_smallerModel_{}rnn_size1_layer'.format(rnn_size)
#     smallerModelPath = os.path.join(integ_dataset_saved_models_folder, model_name)
#     smallerModel = tmm.create_model(transl_dataset.fields, options=smallerOptions)
#     if USE_CUDA: smallerModel = smallerModel.cuda()
#     if model_name not in onmtManager.saved_models:
#         onmtManager.add_new_model(model_name, smallerModelPath,
#                                      arguments_creator_function=smallerOptions)
#     if TRAIN_SMALLER_MODEL:
#         onmtManager.train_model(smallerModel, model_name=model_name,
#                                    train_function=tmm.train_model,
#                                    arguments_train_function={'options':smallerOptions},
#                                    train_loader=train_loader, test_loader=test_loader)
#     if onmtManager.get_num_training_runs(model_name) > 0:
#         smallerModel.load_state_dict(onmtManager.load_model_state_dict(model_name))
#     del smallerModel
#
#     #Distilled model with word-level knowledge distillation
#     teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
#     if USE_CUDA: teacherModel = teacherModel.cuda()
#     teacherModel.load_state_dict(onmtManager.load_model_state_dict(teacherModel_name))
#
#     distilledOptions = copy.deepcopy(smallerOptions)
#     distilled_model_name = 'integ_dataset_distilledModel_word_level_{}rnn_size1_layer'.format(rnn_size)
#     distilled_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
#     if USE_CUDA: distilled_model_word_level = distilled_model_word_level.cuda()
#     distilledModelPath = os.path.join(integ_dataset_saved_models_folder, distilled_model_name)
#     if distilled_model_name not in onmtManager.saved_models:
#         onmtManager.add_new_model(distilled_model_name, distilledModelPath,
#                                      arguments_creator_function=distilledOptions)
#     if TRAIN_WORD_DISTILLED:
#         onmtManager.train_model(distilled_model_word_level, model_name=distilled_model_name,
#                                    train_function=tmm.train_model,
#                                    arguments_train_function={'options':distilledOptions,
#                                                              'teacher_model': teacherModel,
#                                                              'use_distillation_loss':True},
#                                    train_loader=train_loader, test_loader=test_loader)
#     if onmtManager.get_num_training_runs(distilled_model_name) > 0:
#         distilled_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
#     del distilled_model_word_level
#
#     #Quantized word level distillation
#     for numBit in numBits:
#         distilled_model_name_quantized = 'integ_dataset_distilledModel_word_level_quantized{}bits{}rnn_size1_layer'.format(
#                                                                                         numBit, rnn_size)
#         distilled_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
#         if USE_CUDA: distilled_model_word_level = distilled_model_word_level.cuda()
#         distilledModelPath = os.path.join(integ_dataset_saved_models_folder, distilled_model_name_quantized)
#         if distilled_model_name_quantized not in onmtManager.saved_models:
#             onmtManager.add_new_model(distilled_model_name_quantized, distilledModelPath,
#                                          arguments_creator_function=distilledOptions)
#         if TRAIN_WORD_DISTILLED and TRAIN_QUANTIZED_DISTILLED:
#             onmtManager.train_model(distilled_model_word_level, model_name=distilled_model_name_quantized,
#                                        train_function=tmm.train_model,
#                                        arguments_train_function={'options':distilledOptions,
#                                                                  'teacher_model': teacherModel,
#                                                                  'use_distillation_loss':True,
#                                                                  'quantizeWeights':True,
#                                                                  'numBits':numBit,
#                                                                  'bucket_size':256},
#                                        train_loader=train_loader, test_loader=test_loader)
#         if onmtManager.get_num_training_runs(distilled_model_name_quantized) > 0:
#             distilled_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name_quantized))
#         del distilled_model_word_level
#
#         #optimize quantization points
#         if numBit == 8:#but no 8 bits with differentiable quantization
#             continue
#
#         optQuanPointOptions = copy.deepcopy(onmt.standard_options.stdOptions)
#         optQuanPointOptions['learning_rate'] = 1e-4
#         optQuanPointOptions['epochs'] = 3
#         learning_rate_str = str(optQuanPointOptions['learning_rate'])
#         save_path = onmtManager.get_model_base_path(distilled_model_name) + \
#                     'quant_points_{}bit_bucket_size256'.format(numBit)
#         if TRAIN_DIFFERENTIABLE_QUANTIZATION:
#             distilledModel = tmm.create_model(transl_dataset.fields, options=distilledOptions)
#             distilledModel.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
#             if USE_CUDA: distilledModel = distilledModel.cuda()
#             points, infoDict = tmm.optimize_quantization_points(distilledModel, train_loader, test_loader,
#                                                                 optQuanPointOptions, numPointsPerTensor=2**numBit,
#                                                                 bucket_size=256, assignBitsAutomatically=True,
#                                                                 use_distillation_loss=True)
#             quantization_points = [x.data.view(1, -1).cpu().numpy().tolist()[0] for x in points]
#             with open(save_path, 'wb') as p:
#                 pickle.dump((quantization_points, infoDict), p)

#print bleu for the models
example_translations=False
file_results = 'results_file_BLEU_models'
if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
    with open(file_results, 'a') as fr:
        fr.write('\n\n== New Testing Run == \n\n')

if COMPUTE_WORD_PERCENTAGE_SIMILARITY:
    #we need the ref file with the teacher
    teacherModelOptions = onmtManager.load_metadata('integ_dataset_teacherModel', 0)[0]
    for key, val in teacherModelOptions.items(): #remeding to an old bug in save_metadata function
        if val == 'None':
            teacherModelOptions[key] = None
    teacherModel = tmm.create_model(transl_dataset.fields, options=teacherModelOptions)
    if USE_CUDA: teacherModel = teacherModel.cuda()
    teacherModel.load_state_dict(onmtManager.load_model_state_dict('integ_dataset_teacherModel', 1))
    pathTeacherTranslation = transl_hf.get_translation_file_model(teacherModel, transl_dataset,
                                                                  teacherModelOptions, standardTranslateOptions)

for x in onmtManager.list_models():
    if onmtManager.get_num_training_runs(x) == 0:
        continue
    modelOptions = onmtManager.load_metadata(x, 0)[0]
    for key, val in modelOptions.items(): #remeding to an old bug in save_metadata function
        if val == 'None':
            modelOptions[key] = None

    if 'distilled' in x and 'word' not in x:
        dataset = distilled_dataset
    else:
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

    if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
        if COMPUTE_WORD_PERCENTAGE_SIMILARITY is False or (COMPUTE_WORD_PERCENTAGE_SIMILARITY and x != 'integ_dataset_teacherModel'):
            file_translation_model = transl_hf.get_translation_file_model(model, dataset,
                                                                          modelOptions, standardTranslateOptions)
        else:
            file_translation_model = pathTeacherTranslation
        if COMPUTE_BLEU_MODELS:
            bleu = transl_hf.get_bleu_moses(file_translation_model, dataset.testFilesPath[1], file_input=True)
        else:
            bleu = 'Not computed'
        if COMPUTE_WORD_PERCENTAGE_SIMILARITY and x != 'integ_dataset_teacherModel':
            percentage_word_similarity = transl_hf.compute_percentage_word_similarity(pathTeacherTranslation,
                                                                                     file_translation_model,
                                                                                     file_input=True)
        else:
            percentage_word_similarity = 'not computed'
    else:
        bleu = 'Not computed'
        percentage_word_similarity = 'not computed'

    perplexity = onmtManager.load_metadata(x,1)[1]['perplexity'][-1]
    str_to_save = 'Model "{}"  ==> Perplexity: {}, BLEU: {} Percentage word similarity with teacher: {}'.format(x,
                                                                                        perplexity,
                                                                                        bleu,
                                                                                        percentage_word_similarity)
    if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
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

    if CHECK_PM_QUANTIZATION:
        if 'distilledModel_word_level' in x and 'quantized' not in x:
            for numBit in numBits:
                model.load_state_dict(onmtManager.load_model_state_dict(x, 1))
                for p in model.parameters():
                    p.data = quantization.uniformQuantization(p.data, s=2**numBit, type_of_scaling='linear',
                                                              bucket_size=256)[0]
                perplexity = tmm.evaluate_model(model, test_loader).ppl()

                if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
                    file_translation_model = transl_hf.get_translation_file_model(model, dataset,
                                                                                  modelOptions,
                                                                                  standardTranslateOptions)
                    if COMPUTE_BLEU_MODELS:
                        bleu = transl_hf.get_bleu_moses(file_translation_model, dataset.testFilesPath[1],
                                                        file_input=True)
                    else:
                        bleu = 'Not computed'
                    if COMPUTE_WORD_PERCENTAGE_SIMILARITY:
                        percentage_word_similarity = transl_hf.compute_percentage_word_similarity(
                            pathTeacherTranslation,
                            file_translation_model,
                            file_input=True)
                    else:
                        percentage_word_similarity = 'not computed'
                else:
                    bleu = 'Not computed'
                    percentage_word_similarity = 'not computed'


                str_to_save = 'PM quantization of model "{}" with "{}" bits and bucket size 256: Perplexity : {}, BLEU: {}'.format(
                    x, numBit, perplexity, bleu)
                str_to_save += 'Percentage word similarity with teacher:{}'.format(percentage_word_similarity)
                if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
                    with open(file_results, 'a') as fr:
                        fr.write(str_to_save + '\n')
                print(str_to_save)
                quant_fun = functools.partial(quantization.uniformQuantization, s=2**numBit, bucket_size=256)
                actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                               'uniform', s=2**numBit)
                print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                                                              mhf.get_size_reduction(
                                                                                  actual_bit_huffmman,
                                                                                  bucket_size=256)))
#now for the models trained with the differentiable quantization algorithm
list_distilled_models = ['integ_dataset_distilledModel_word_level_{}rnn_size1_layer'.format(x)
                         for x in rnn_sizes]
optQuanPointOptions = copy.deepcopy(onmt.onmt.standard_options.stdOptions)
for idx_model_distilled, distilled_model_name_to_quantize in enumerate(list_distilled_models):
    modelOptions = onmtManager.load_metadata(distilled_model_name_to_quantize, 0)[0]
    for key, val in modelOptions.items():  # remeding to an old bug in save_metadata function
        if val == 'None':
            modelOptions[key] = None
    dataset = transl_dataset #since we don't use sequence level distillation
    for numBit in numBits:
        if numBit == 8: continue
        save_path = onmtManager.get_model_base_path(distilled_model_name_to_quantize) + \
                    'quant_points_{}bit_bucket_size256'.format(numBit)
        with open(save_path, 'rb') as p:
            quantization_points, infoDict = pickle.load(p)
        distilledModel = tmm.create_model(dataset.fields, options=modelOptions)
        distilledModel.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name_to_quantize))
        if USE_CUDA: distilledModel = distilledModel.cuda()
        for idx, p in enumerate(distilledModel.parameters()):
            p.data = quantization.nonUniformQuantization(p.data, quantization_points[idx], bucket_size=256)[0]
        reported_perplexity = infoDict['perplexity'][-1]
        perplexity = tmm.evaluate_model(distilledModel, test_loader).ppl()

        if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
            file_translation_model = transl_hf.get_translation_file_model(distilledModel, dataset,
                                                                          modelOptions,
                                                                          standardTranslateOptions)
            if COMPUTE_BLEU_MODELS:
                bleu = transl_hf.get_bleu_moses(file_translation_model, dataset.testFilesPath[1],
                                                file_input=True)
            else:
                bleu = 'Not computed'
            if COMPUTE_WORD_PERCENTAGE_SIMILARITY:
                percentage_word_similarity = transl_hf.compute_percentage_word_similarity(
                    pathTeacherTranslation,
                    file_translation_model,
                    file_input=True)
            else:
                percentage_word_similarity = 'not computed'
        else:
            bleu = 'Not computed'
            percentage_word_similarity = 'not computed'

        str_to_save = 'Model "{}"  ==> Reported perplexity : {}, Actual perplexity: {}, BLEU: {}'.format(
            distilled_model_name_to_quantize + 'quant_points_{}bit_bucket_size256'.format(numBit),
            reported_perplexity, perplexity, bleu)
        str_to_save += 'Percentage word similarity with teacher:{}'.format(percentage_word_similarity)
        if COMPUTE_BLEU_MODELS or COMPUTE_WORD_PERCENTAGE_SIMILARITY:
            with open(file_results, 'a') as fr:
                fr.write(str_to_save + '\n')
        print(str_to_save)

        quantization_functions = [functools.partial(quantization.nonUniformQuantization,
                                                    listQuantizationPoints=qp,
                                                    bucket_size=256) for qp in quantization_points]
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(distilledModel.parameters(),
                                                                       quantization_functions,
                                                                       'nonUniform')
        print('Effective bit Huffman: {} - Size reduction: {}'.format(actual_bit_huffmman,
                                                                      mhf.get_size_reduction(
                                                                          actual_bit_huffmman,
                                                                          bucket_size=256)))


try:
    os.remove(pathTeacherTranslation)
except:pass
try:
    os.remove(file_translation_model)
except:pass
