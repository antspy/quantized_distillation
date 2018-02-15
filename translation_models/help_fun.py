import onmt
import onmt.Models
import onmt.modules
import onmt.Trainer
import onmt.IO
import helpers.functions as mhf
import shutil
import os
import subprocess
import translation_models
import re
import math
from itertools import takewhile, count
from itertools import zip_longest
import torch
import copy
import onmt.standard_options
import random
import uuid


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


class MyTrainer(onmt.Trainer):

    def forward_and_backward(self, idx_batch, batch, epoch, total_stats, report_func,
                             use_distillation_loss=False, teacher_model=None):

        if use_distillation_loss is True and teacher_model is None:
            raise ValueError('To compute distillation loss you need to pass the teacher model')

        report_stats = onmt.Statistics()
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size

        dec_state, dec_state_teacher = None, None
        _, src_lengths = batch.src

        src = onmt.IO.make_features(batch, 'src')
        tgt_outer = onmt.IO.make_features(batch, 'tgt')
        report_stats.n_src_words += src_lengths.sum()

        for j in range(0, target_size - 1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            outputs, attns, dec_state = self.model(src, tgt, src_lengths, dec_state)
            if use_distillation_loss:
                teacher_outputs, _, dec_state_teacher = teacher_model(src, tgt, src_lengths, dec_state_teacher)
                teacher_outputs = teacher_outputs.detach()
                if dec_state_teacher is not None:
                    dec_state_teacher = dec_state_teacher.detach()
            else:
                teacher_outputs = None

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = self.train_loss.sharded_compute_loss(batch, outputs, attns, j, trunc_size, self.shard_size,
                                                               teacher_outputs)

            # 4. Update the statistics.
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        if report_func is not None:
            report_func(epoch, idx_batch, len(self.train_iter),
                        total_stats.start_time, self.optim.lr,
                        report_stats)
        return total_stats


class Translator_modified(onmt.Translator):
    def __init__(self, model, fields, model_options, translate_options):
        # Add in default model arguments, possibly added since training.
        # Note: The fields here should contain src_map, so it is not what the translation_dataset loader does
        # as it removes those fields. Be careful
        self.opt = mhf.convertToNamedTuple(translate_options)
        self.fields = fields
        model_opt = model_options
        for arg in translate_options:
            if arg not in model_opt:
                model_opt[arg] = translate_options[arg]

        model_opt = mhf.convertToNamedTuple(model_opt)

        self._type = model_opt.model_type
        self.copy_attn = model_opt.copy_attn

        self.model = model
        self.model.eval()
        self.model.generator.eval()

        # for debugging
        self.beam_accum = None

def translate_sequences(model, vocab_file, model_options, translate_options, source_to_translate, out_file_path=None,
                        verbose=True, percentages_to_show=25, replace_unk_hack=False):

    'translates phrases using the model passed as parameter'

    fields = onmt.IO.ONMTDataset.load_fields(torch.load(vocab_file))
    translator = Translator_modified(model, fields, model_options, translate_options)
    for key, val in onmt.standard_options.standardPreProcessingOptions.items():
        if key not in translate_options:
            translate_options[key] = val

    translate_options = mhf.convertToNamedTuple(translate_options)
    model_options = mhf.convertToNamedTuple(model_options)

    iterator_created = False
    dataset_created = False
    is_source_file = False
    #source_to_translate can be different things:
    # - A ONMTDataset (must have been created with options=None though!)
    # - An OrderedIterator (must also have been created with options=None)
    # - A path to a file
    # - A string of phrases separated by \n
    # - A list of strings

    if isinstance(source_to_translate, onmt.IO.ONMTDataset):
        dataset = source_to_translate
        dataset_created = True
    elif isinstance(source_to_translate, onmt.IO.OrderedIterator):
        data_generator = source_to_translate
        dataset = source_to_translate.dataset
        dataset_created = True
        iterator_created = True
    elif isinstance(source_to_translate, str):
        if os.path.exists(source_to_translate):
            dataset = onmt.IO.ONMTDataset(source_to_translate, source_to_translate, translator.fields, None)
            dataset_created = True
            is_source_file, source_file_path = True, source_to_translate
        else:
            source_to_translate = re.sub('\n+', '\n', source_to_translate)
            source_to_translate = [x for x in source_to_translate.split('\n') if x]

    if isinstance(source_to_translate, list):
        temp_file_path = os.path.abspath('temp_file_translate_pytorch_{}'.format(uuid.uuid4()))
        with open(temp_file_path, 'w') as temp_file:
            for line in source_to_translate:
                temp_file.write(line + '\n')
        is_source_file, source_file_path = True, temp_file_path
        dataset = onmt.IO.ONMTDataset(temp_file_path, temp_file_path, translator.fields, None)
        dataset_created = True

    if not dataset_created:
        raise ValueError('source_to_translate could not have been interpreted correctly')

    if not iterator_created:
        data_generator = onmt.IO.OrderedIterator(dataset=dataset, device=model_options.gpu,
                                                 batch_size=translate_options.batch_size,
                                                 train=False, sort=False, shuffle=False)

    if out_file_path is None:
        res = ''
    else:
        out_file = open(out_file_path, 'w')

    next_percentage_to_show = percentages_to_show
    total_num_batches = len(data_generator)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    counter = count(1)

    if replace_unk_hack and is_source_file:
        if data_generator.batch_size != 1:
            raise ValueError('For now the replace_unk_hack only works with batch size 1')
        source_file = open(source_file_path, 'r')
        lines_iterator = (line for line in source_file)

    try:
        for idx, batch in enumerate(data_generator):
            pred_batch, gold_batch, pred_scores, gold_scores, attn, src = translator.translate(batch, dataset)
            pred_score_total += sum(score[0] for score in pred_scores)
            pred_words_total += sum(len(x[0]) for x in pred_batch)
            if translate_options.tgt:
                gold_score_total += sum(gold_scores)
                gold_words_total += sum(len(x) for x in batch.tgt[1:])

            # z_batch: an iterator over the predictions, their scores,
            # the gold sentence, its score, and the source sentence for each
            # sentence in the batch. It has to be zip_longest instead of
            # plain-old zip because the gold_batch has length 0 if the target
            # is not included.
            z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

            for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
                n_best_preds = [pred for pred in pred_sents[:translate_options.n_best]]
                if replace_unk_hack and is_source_file:
                    original_line = next(lines_iterator).split(' ')
                    for pred in n_best_preds:
                        for idx_tok, tok in enumerate(pred):
                            if tok == '<unk>':
                                _, maxIndex = attn[0][0][idx_tok].max(0)
                                pred[idx_tok] = original_line[maxIndex[0]]
                n_best_preds = [" ".join(pred) for pred in n_best_preds]
                strToWrite = '\n'.join(n_best_preds) + '\n'

                if out_file_path is None:
                    res += strToWrite
                else:
                    out_file.write(strToWrite)
                    out_file.flush()

                if translate_options.verbose:
                    sent_number = next(counter)
                    words = get_src_words(
                        src_sent, translator.fields["src"].vocab.itos)

                    os.write(1, bytes('\nSENT %d: %s\n' %
                                      (sent_number, words), 'UTF-8'))

                    best_pred = n_best_preds[0]
                    best_score = pred_score[0]
                    os.write(1, bytes('PRED %d: %s\n' %
                                      (sent_number, best_pred), 'UTF-8'))
                    print("PRED SCORE: %.4f" % best_score)

                    if translate_options.tgt:
                        tgt_sent = ' '.join(gold_sent)
                        os.write(1, bytes('GOLD %d: %s\n' %
                                          (sent_number, tgt_sent), 'UTF-8'))
                        print("GOLD SCORE: %.4f" % gold_score)

                    if len(n_best_preds) > 1:
                        print('\nBEST HYP:')
                        for score, sent in zip(pred_score, n_best_preds):
                            os.write(1, bytes("[%.4f] %s\n" % (score, sent), 'UTF-8'))

            if idx / total_num_batches * 100 >= next_percentage_to_show:
                if verbose:
                    print('Total completed: {:.2f}%'.format(idx / total_num_batches * 100))
                next_percentage_to_show += percentages_to_show

        #report_score('PRED', pred_score_total, pred_words_total)

        if translate_options.tgt:
            report_score('GOLD', gold_score_total, gold_words_total)

    except Exception as e:
        print('An error occurred in translating sentences: {}'.format(e))
    try:
        source_file.close()
    except:pass
    try:
        os.remove(temp_file_path)
    except:pass

    if out_file_path is None:
        return res
    else:
        out_file.close()
        return out_file_path

def create_distillation_dataset(model, model_options, translate_options, original_dataset, folder_to_save,
                                path_moses_tokenizer=None, tokenize_out_file=False):

    '''
    creates a distillation dataset to use with a student model, according to the paper
    "Sequence-level knowledge distillation". We use as target sentence not the real sentences, but the senteces
    translated by the model. So the new dataset will be equal to the old one, except the target sentences are the one
    given in output by the model.
    '''

    if path_moses_tokenizer is None:
        perl_tokenizer = os.path.join(translation_models.PATH_PERL_SCRIPTS_FOLDER, 'tokenizer.perl')
    else:
        perl_tokenizer = path_moses_tokenizer

    try:
        os.mkdir(folder_to_save)
    except:pass

    shutil.copy2(original_dataset.trainFilesPath[0], folder_to_save)
    shutil.copy2(original_dataset.testFilesPath[0], folder_to_save)
    shutil.copy2(original_dataset.testFilesPath[1], folder_to_save)

    out_file_path = os.path.join(folder_to_save, os.path.basename(original_dataset.trainFilesPath[1]))
    out_file_path_untoken = out_file_path + '.atok'
    translate_sequences(model, original_dataset.processedFilesPath[0], model_options, translate_options,
                                source_to_translate=original_dataset.trainFilesPath[0],
                                out_file_path=out_file_path_untoken)

    if tokenize_out_file:
        #finally, use moses tokenizer on the out file
        with open(out_file_path_untoken, 'r') as untokenFile, open(out_file_path, 'w') as tokenFile:
            subprocess.call(['perl', perl_tokenizer, '-a', '-q', '-no-escape', '-threads', '2', '-l',
                             original_dataset.tgt_language], stdin=untokenFile, stdout=tokenFile)
        try:
            os.remove(out_file_path_untoken)
        except:pass
    else:
        os.rename(out_file_path_untoken, out_file_path)

def get_bleu_moses(hypotheses, reference, file_input=False, path_to_multi_bleu_moses=None):

    '''
    Get BLEU score with moses bleu score. if file_input is False, then hypothesis and reference
    are supposed to be list of strings. If file_input is true, then they are supposed to be path
    to files with one sentence per line.
    '''

    if path_to_multi_bleu_moses is None:
        bleu_perl_script = os.path.join(translation_models.PATH_PERL_SCRIPTS_FOLDER, 'multi-bleu.perl')
    else:
        bleu_perl_script = path_to_multi_bleu_moses

    if file_input is True:
        with open(hypotheses, 'r') as hypothesis_file:
            pipe = subprocess.Popen(
                ["perl", bleu_perl_script, '-lc', reference],
                stdin=hypothesis_file,
                stdout=subprocess.PIPE)
    else:
        temp_file = os.path.abspath('tmp_reference' + str(uuid.uuid4()) + '.txt')
        with open(temp_file, 'w') as f:
            for ref in reference:
                f.write(ref + '\n')

        hypothesis_pipe = '\n'.join(hypotheses)
        hypothesis_pipe = hypothesis_pipe.encode()
        pipe = subprocess.Popen(
            ["perl", bleu_perl_script, '-lc', temp_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        pipe.stdin.write(hypothesis_pipe)
        pipe.stdin.close()

        #clean up temp files
        try:
            os.remove(temp_file)
        except:pass

    resultBleu = pipe.stdout.read()
    return resultBleu


def get_translation_file_model(model, dataset, options_model, options_translate=None,
                          percentages_to_show=25, verbose=False, hypothesis_file_path=None):

    if options_translate is None:
        options_translate = copy.deepcopy(onmt.standardOptions.standardTranslationOptions)

    if hypothesis_file_path is None:
        hypothesis_file_path = os.path.abspath(str(uuid.uuid4()))  # random name, no conflicts

    translate_sequences(model, dataset.processedFilesPath[0], options_model, options_translate,
                        source_to_translate=dataset.testFilesPath[0], out_file_path=hypothesis_file_path,
                        percentages_to_show=percentages_to_show, verbose=verbose)
    return hypothesis_file_path


def get_bleu_model(model, dataset, options_model, options_translate=None, percentages_to_show=25, verbose=False):

    'get bleu scores given a model and a dataset. it uses get_bleu_moses to compute the scores'

    reference_file = dataset.testFilesPath[1]
    hypothesis_file = get_translation_file_model(model, dataset, options_model, options_translate=options_translate,
                                            percentages_to_show=percentages_to_show, verbose=verbose)
    bleu = get_bleu_moses(hypothesis_file, reference_file, file_input=True)
    os.remove(hypothesis_file)

    return bleu


def get_translation_examples(model, dataset, num_examples, options_model,
                             options_translate=None, approx_length_file=1000, shuffle_examples=True):

    if options_translate is None:
        options_translate = copy.deepcopy(onmt.standardOptions.standardTranslationOptions)

    linesToTranslate = ''
    referenceLines = ''
    #it is not uniformly random, but it doesn't really matter, this is just to get some examples.
    if shuffle_examples:
        start_line = random.randrange(approx_length_file//2)
    else:
        start_line = 30
    with open(dataset.testFilesPath[0], 'r') as pDE, open(dataset.testFilesPath[1], 'r') as pEN:
        for idx, (lineDE, lineEN) in enumerate(zip(pDE, pEN)):
            if num_examples == 0:
                break
            if idx < start_line:
                continue
            if shuffle_examples:
                if random.randrange(100) == 0:
                    take_line = True
                else:
                    take_line = False
            else:
                take_line = True
            if take_line:
                linesToTranslate += lineDE
                referenceLines += lineEN
                num_examples -= 1

    translated_lines = translate_sequences(model, dataset.processedFilesPath[0], options_model,
                                           options_translate, source_to_translate=linesToTranslate,
                                            out_file_path=None, verbose=False)
    linesToTranslate, translated_lines, referenceLines = [x.split('\n')[:-1] for x in
                                                          (linesToTranslate, translated_lines, referenceLines)]
    return linesToTranslate, translated_lines, referenceLines

def compute_percentage_word_similarity(teacher_translations, model_translation, file_input=False):

    if file_input is not True:
        lines_iter_teacher = teacher_translations.split('\n')
        lines_iter_model = model_translation.split('\n')
    else:
        lines_iter_teacher = open(teacher_translations, 'r')
        lines_iter_model = open(model_translation, 'r')

    sum_percentages = 0
    count_percentages = 0

    for lineTeacher, lineModel in zip(lines_iter_teacher, lines_iter_model):
        words_teacher = lineTeacher.split(' ')
        words_model = lineModel.split(' ')
        n = max(len(words_teacher), len(words_model)) #use max number of words to compute percentage
        count = 0
        for idx in range(min(len(words_teacher), len(words_model))):
            if words_teacher[idx].lower() == words_model[idx].lower():
                count += 1
        sum_percentages += count/n
        count_percentages += 1

    return sum_percentages/count_percentages
