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

print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'
USE_CUDA = torch.cuda.is_available()

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

epochsToTrainOnmtIntegDataset = 13

onmtManager = model_manager.ModelManager('model_manager_multi30k_dataset.tst',
                                         'model_manager', create_new_model_manager=False)
for x in onmtManager.list_models():
    if onmtManager.get_num_training_runs(x) > 0:
        print(x, onmtManager.load_metadata(x)[1]['perplexity'][-1])


multi30k_saved_models_folder = os.path.join(SAVED_MODELS_FOLDER, 'multi30k')
try:
    os.mkdir(multi30k_saved_models_folder)
except:pass


#load the data
batch_size = 256
transl_dataset = datasets.multi30k_DE_EN(pin_memory=True)
train_loader, test_loader = transl_dataset.getTrainLoader(batch_size), transl_dataset.getTestLoader(batch_size)

#Teacher model
#Like in the paper "Sequence level knowledge distillation", teacher model on de-en translation is 4 layer LSTM with
#hidden dimension 1000
teacherOptions = copy.deepcopy(onmt.standard_options.stdOptions)
teacherOptions['batch_size'] = batch_size #it only matter in the creation of the distillation dataset
teacherOptions['rnn_size'] = 500
teacherOptions['epochs'] = epochsToTrainOnmtIntegDataset
model_name = 'multi30k_teacherModelNewCodebase_v2'
teacherModelPath = os.path.join(multi30k_saved_models_folder, model_name)
teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
if USE_CUDA: teacherModel = teacherModel.cuda()
if model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(model_name, teacherModelPath,
                                 arguments_creator_function=teacherOptions)
# for _ in range(5):
#     onmtManager.train_model(teacherModel, model_name=model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':teacherOptions},
#                                train_loader=train_loader, test_loader=test_loader)
teacherModel.load_state_dict(onmtManager.load_model_state_dict(model_name))

#now create a distillation dataset
create_distilled_dataset_options = copy.deepcopy(teacherOptions)
create_distilled_dataset_options['batch_size'] = 1
print('Checkpoint: distilledModel openNMT integration dataset\n')
standardTranslateOptions = onmt.standard_options.standardTranslationOptions
folder_distillation_dataset = os.path.join(transl_dataset.dataFolder, 'distilled_dataset_' + model_name)
# transl_hf.create_distillation_dataset(teacherModel, create_distilled_dataset_options, standardTranslateOptions,
#                                         transl_dataset, folder_distillation_dataset)
distilled_dataset = datasets.translation_datasets.TranslationDataset(folder_distillation_dataset, src_language='de',
                                                                     tgt_language='en', pin_memory=True)
train_distilled_loader, test_distilled_loader = distilled_dataset.getTrainLoader(batch_size), distilled_dataset.getTestLoader(batch_size)
print('Distillation dataset created')

#create and train the smaller model
print('Checkpoint: smallerModel openNMT integration dataset\n')
smallerOptions = copy.deepcopy(onmt.standard_options.stdOptions)
smallerOptions['rnn_size'] = 100
smallerOptions['layers'] = 1
smallerOptions['enc_layers'] = 1 #these two parameters seem to have no effect...
smallerOptions['dec_layers'] = 1
smallerOptions['epochs'] = epochsToTrainOnmtIntegDataset

smaller_model_name = 'multi30k_smallerModelNewCodebase_v2'
smaller_model = tmm.create_model(transl_dataset.fields, options=smallerOptions)
if USE_CUDA: smaller_model = smaller_model.cuda()
smallerModelPath = os.path.join(multi30k_saved_models_folder, smaller_model_name)

if smaller_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(smaller_model_name, smallerModelPath,
                                 arguments_creator_function=smallerOptions)
# for _ in range(5):
#     onmtManager.train_model(smaller_model, model_name=smaller_model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':smallerOptions},
#                                train_loader=train_loader, test_loader=test_loader)
smaller_model.load_state_dict(onmtManager.load_model_state_dict(smaller_model_name))

del smaller_model

#create and train the distilled model with sequence-level knowledge distillation
distilledOptions = copy.deepcopy(smallerOptions)
distilledOptions['epochs'] = 20

distilled_model_name = 'multi30k_distilledModelNewCodebase_v2'
distilled_model = tmm.create_model(distilled_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_model = distilled_model.cuda()
distilledModelPath = os.path.join(multi30k_saved_models_folder, distilled_model_name)
if distilled_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
# for _ in range(5):
#     onmtManager.train_model(distilled_model, model_name=distilled_model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':distilledOptions},
#                                train_loader=train_distilled_loader, test_loader=test_distilled_loader)
distilled_model.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
del distilled_model

#quantized sequence level distillation
distilled_model_name = 'multi30k_distilled_quantizedModelNewCodebase_v2'
distilled_model = tmm.create_model(distilled_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_model = distilled_model.cuda()
distilledModelPath = os.path.join(multi30k_saved_models_folder, distilled_model_name)
if distilled_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
# for _ in range(5):
#     onmtManager.train_model(distilled_model, model_name=distilled_model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':distilledOptions,
#                                                          'quantizeWeights': True,
#                                                          'numBits': 4,
#                                                          'bucket_size': 256},
#                                train_loader=train_distilled_loader, test_loader=test_distilled_loader)
distilled_model.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
del distilled_model



#create and train the distilled model with word-level knowledge distillation
#
distilled_model_name = 'multi30k_distilledModel_word_levelNewCodebase_v2'
distilled_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_model_word_level = distilled_model_word_level.cuda()
distilledModelPath = os.path.join(multi30k_saved_models_folder, distilled_model_name)
if distilled_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
# for _ in range(5):
#     onmtManager.train_model(distilled_model_word_level, model_name=distilled_model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':distilledOptions,
#                                                          'teacher_model': teacherModel,
#                                                          'use_distillation_loss':True},
#                                train_loader=train_loader, test_loader=test_loader)
distilled_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
del distilled_model_word_level

#Now quantized word level distillation
distilled_model_name = 'multi30k_distilled_quantizedModel_word_levelNewCodebase_v2'
distilled_model_word_level = tmm.create_model(transl_dataset.fields, options=distilledOptions)
if USE_CUDA: distilled_model_word_level = distilled_model_word_level.cuda()
distilledModelPath = os.path.join(multi30k_saved_models_folder, distilled_model_name)
if distilled_model_name not in onmtManager.saved_models:
    onmtManager.add_new_model(distilled_model_name, distilledModelPath,
                                 arguments_creator_function=distilledOptions)
# for _ in range(5):
#     onmtManager.train_model(distilled_model_word_level, model_name=distilled_model_name,
#                                train_function=tmm.train_model,
#                                arguments_train_function={'options':distilledOptions,
#                                                          'teacher_model': teacherModel,
#                                                          'use_distillation_loss':True,
#                                                          'quantizeWeights':True,
#                                                          'numBits':4,
#                                                          'bucket_size':256},
#                                train_loader=train_loader, test_loader=test_loader)
distilled_model_word_level.load_state_dict(onmtManager.load_model_state_dict(distilled_model_name))
del distilled_model_word_level
del teacherModel


#print model perplexity and BLEU
for x in onmtManager.list_models():
    perplexity = onmtManager.load_metadata(x)[1]['perplexity'][-1]
    if '_v2' not in x:continue
    if 'distilled_quantized' in x:
        to_quantize=True
    else:to_quantize=False
    # get bleu
    if 'distilled' in x and 'word' not in x:
        dataset = distilled_dataset
    else:
        dataset = transl_dataset

    if 'distilled' in x and 'word' not in x:
        currOptions = distilledOptions
    elif 'teacher' in x:
        currOptions = teacherOptions
    elif 'word_level' in x:
        currOptions = distilledOptions
    else:
        currOptions = smallerOptions

    currOptions['batch_size'] = 1 #important for the BLEU computation.
    model = tmm.create_model(dataset.fields, options=currOptions)
    if USE_CUDA: model = model.cuda()
    model.load_state_dict(onmtManager.load_model_state_dict(x))
    if to_quantize:
        for p in model.parameters():
            p.data = quantization.uniformQuantization(p.data, 2**4, bucket_size=256)[0]

    num_examples = 5
    print('Example of translation for "{}"'.format(x))
    linesToTranslate, translated_lines, referenceLines = transl_hf.get_translation_examples(model, dataset, num_examples,
                                                                            currOptions, standardTranslateOptions)
    print('Original Sentences == Translation == Ref Translation')
    print('\n'.join(' == '.join(x) for x in zip(linesToTranslate, translated_lines, referenceLines)))
    bleu = transl_hf.get_bleu_model(model, dataset, currOptions, standardTranslateOptions)
    print('Model "{}"  ==> Perplexity: {}, BLEU: {}'.format(x, perplexity, bleu))


## Try naive quantization on teacher model
numBits = [8, 4, 2]
bucket_sizes = [256]

# for numBit in numBits:
#     for bucket_size in bucket_sizes:
#         teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
#         teacherModel.load_state_dict(onmtManager.load_model_state_dict('multi30k_teacherModelNewCodebase_v2'))
#         if USE_CUDA: teacherModel = teacherModel.cuda()
#         for p in teacherModel.parameters():
#             p.data = quantization.uniformQuantization(p.data, 2**numBit, bucket_size=bucket_size)[0]
#         perplexity = tmm.evaluate_model(teacherModel, test_loader).ppl()
#         options = copy.deepcopy(teacherOptions)
#         options['batch_size'] = 1
#         bleu = transl_hf.get_bleu_model(teacherModel, transl_dataset, options, standardTranslateOptions)
#         print('teacher model {} bits {} bucket size : {} perplexity, BLEU:{}'.format(numBit, bucket_size,
#                                                                                      perplexity, bleu))

#optimize quantization points
numBits = [4, 2]
bucket_sizes = [256]
optQuanPointOptions = copy.deepcopy(onmt.standard_options.stdOptions)
optQuanPointOptions['learning_rate'] = 1e-5
optQuanPointOptions['epochs'] = 10
learning_rate_str = str(optQuanPointOptions['learning_rate'])

# for numBit in numBits:
#     for bucket_size in bucket_sizes:
#         teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
#         teacherModel.load_state_dict(onmtManager.load_model_state_dict('multi30k_teacherModelNewCodebase_v2'))
#         if USE_CUDA: teacherModel = teacherModel.cuda()
#         points, infoDict = tmm.optimize_quantization_points(teacherModel, train_loader, test_loader,
#                                                             optQuanPointOptions, numPointsPerTensor=2**numBit,
#                                                             bucket_size=bucket_size,
#                                                             assignBitsAutomatically=True,
#                                                             use_distillation_loss=True)
#         quantization_points = [x.data.view(1, -1).cpu().numpy().tolist()[0] for x in points]
#         save_path = onmtManager.get_model_base_path('multi30k_teacherModelNewCodebase_v2') + \
#                                 'quant_points_{}bits{}bucketLr{}'.format(numBit, bucket_size, learning_rate_str)
#         with open(save_path, 'wb') as p:
#             pickle.dump((quantization_points, infoDict), p)

optQuanPointOptions['batch_size'] = 1
for bucket_size in bucket_sizes:
    for numBit in numBits:
        save_path = onmtManager.get_model_base_path('multi30k_teacherModelNewCodebase_v2') + \
                    'quant_points_{}bits{}bucketLr{}'.format(numBit, bucket_size, learning_rate_str)
        with open(save_path, 'rb') as p:
            quantization_points, infoDict = pickle.load(p)
        teacherModel = tmm.create_model(transl_dataset.fields, options=teacherOptions)
        teacherModel.load_state_dict(onmtManager.load_model_state_dict('multi30k_teacherModelNewCodebase_v2'))
        for idx, p in enumerate(teacherModel.parameters()):
            p.data = quantization.nonUniformQuantization(p.data, quantization_points[idx], bucket_size=bucket_size)[0]
        if USE_CUDA: teacherModel = teacherModel.cuda()
        perplexity = tmm.evaluate_model(teacherModel, test_loader).ppl()
        bleu = transl_hf.get_bleu_model(teacherModel, transl_dataset, optQuanPointOptions, standardTranslateOptions)
        print('quant_points{}bits{}bucket ==>  perplexity: {}, BLEU:{}'.format(numBit, bucket_size,
                                                                                     perplexity, bleu))
