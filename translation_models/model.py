import onmt
import onmt.modules
import onmt.ModelConstructor
import helpers.functions as mhf
import torch
import torch.nn as nn
import copy
import torch.optim
import translation_models.help_fun as thf
import quantization
import quantization.help_functions as qhf
from torch.autograd import Variable
from onmt.Utils import aeq

USE_CUDA = torch.cuda.is_available()

def handle_options(options_dict):
    options_dict = copy.deepcopy(options_dict)
    ## Dealing with options behavior
    if options_dict['word_vec_size'] != -1:
        options_dict['src_word_vec_size'] = options_dict['word_vec_size']
        options_dict['tgt_word_vec_size'] = options_dict['word_vec_size']

    if options_dict['layers'] != -1:
        options_dict['enc_layers'] = options_dict['layers']
        options_dict['dec_layers'] = options_dict['layers']

        options_dict['brnn'] = (options_dict['encoder_type'] == 'brnn')

    if options_dict['seed'] > 0:
        torch.manual_seed(options_dict['seed'])

    if options_dict['gpuid']:
        torch.cuda.set_device(options_dict['gpuid'][0])
        if options_dict['seed'] > 0:
            torch.cuda.manual_seed(options_dict['seed'])

    return options_dict

def create_model(fields, options=None):
    if options is None: options = copy.deepcopy(onmt.standard_options.stdOptions)
    if not isinstance(options, dict):
        options = mhf.convertToDictionary(options)
    options = handle_options(options)
    options = mhf.convertToNamedTuple(options)
    model = onmt.ModelConstructor.make_base_model(options, fields, USE_CUDA, checkpoint=None)
    if len(options.gpuid) > 1:
        model = nn.DataParallel(model, device_ids=options.gpuid, dim=1)

    return model

def create_optimizer(model_or_iterable, options=None):
    if options is None: options = copy.deepcopy(onmt.standard_options.stdOptions)
    if not isinstance(options, dict):
        options = mhf.convertToDictionary(options)
    options = handle_options(options)
    options = mhf.convertToNamedTuple(options)
    optim = onmt.Optim(
        options.optim, options.learning_rate, options.max_grad_norm,
        lr_decay=options.learning_rate_decay,
        start_decay_at=options.start_decay_at,
        opt=options)

    try:
        optim.set_parameters(model_or_iterable.parameters())
    except AttributeError:
        optim.set_parameters(model_or_iterable)
    return optim

def collect_features(train, fields):
    # TODO: account for target features.
    # Also, why does fields need to have the structure it does?
    src_features = onmt.IO.ONMTDataset.collect_features(fields)
    aeq(len(src_features), train.nfeatures)

    return src_features

def make_loss_compute(model, tgt_vocab, dataset, copy_attn=False,
                      copy_attn_force=None, use_distillation_loss=False, teacher_model=None):

    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute* class, by subclassing LossComputeBase.
    """

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('To compute distillation loss you have to pass the teacher model generator')

    if teacher_model is not None:
        teacher_model_generator = teacher_model.generator
    else:
        teacher_model_generator = None

    if copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(model.generator, tgt_vocab, dataset, copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab, use_distillation_loss, teacher_model_generator)

    if USE_CUDA:
        compute = compute.cuda()

    return compute

def report_func(epoch, batch, num_batches, start_time, lr, report_stats, options=None):

    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): a Statistics instance.
    """

    if options is None:
        report_every = onmt.standard_options.stdOptions['report_every']
    else:
        try:
            report_every = options['report_every']
        except KeyError:
            report_every = options.report_every

    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)

def evaluate_model(model, test_loader, copy_attn=False, copy_attn_force=None):

    """ Copied from the method in onmt.Trainer """

    # Set model in validating mode.
    model.eval()

    stats = onmt.Statistics()
    valid_loss = make_loss_compute(model, test_loader.dataset.fields["tgt"].vocab, test_loader.dataset,
                                   copy_attn=copy_attn, copy_attn_force=copy_attn_force)

    for batch in test_loader:
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt = onmt.IO.make_features(batch, 'tgt')

        # F-prop through the model.
        outputs, attns, _ = model(src, tgt, src_lengths)

        # Compute loss.
        gen_state = onmt.Loss.make_gen_state(
            outputs, batch, attns, (0, batch.tgt.size(0)))
        _, batch_stats = valid_loss(batch, **gen_state)

        # Update statistics.
        stats.update(batch_stats)

    # Set model back to training mode.
    model.train()
    return stats

def train_model(model, train_loader, test_loader, optim=None, options=None,
                stochasticRounding=False, quantizeWeights=False, numBits=8,
                maxElementAllowedForQuantization=False, bucket_size=None,
                subtractMeanInQuantization=False, quantizationFunctionToUse='uniformLinearScaling',
                backprop_quantization_style='none', num_estimate_quant_grad=1,
                use_distillation_loss=False, teacher_model=None,
                quantize_first_and_last_layer=True):

    if options is None: options = copy.deepcopy(onmt.standard_options.stdOptions)
    if not isinstance(options, dict):
        options = mhf.convertToDictionary(options)
    options = handle_options(options)
    options = mhf.convertToNamedTuple(options)

    if optim is None:
        optim = create_optimizer(model, options)

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('If training with distilled word level, we need teacher_model to be passed')

    if teacher_model is not None:
        teacher_model.eval()

    step_since_last_grad_quant_estimation = 0
    num_param_model = sum(1 for _ in model.parameters())
    if quantizeWeights:
        quantizationFunctionToUse = quantizationFunctionToUse.lower()
        if quantizationFunctionToUse == 'uniformAbsMaxScaling'.lower():
            s = 2 ** (numBits - 1)
            type_of_scaling = 'absmax'
        elif quantizationFunctionToUse == 'uniformLinearScaling'.lower():
            s = 2 ** numBits
            type_of_scaling = 'linear'
        else:
            raise ValueError('The specified quantization function is not present')

        if backprop_quantization_style is None or backprop_quantization_style in ('none', 'truncated'):
            quantizeFunctions = lambda x: quantization.uniformQuantization(x, s,
                                                    type_of_scaling=type_of_scaling,
                                                    stochastic_rounding=stochasticRounding,
                                                    max_element=maxElementAllowedForQuantization,
                                                    subtract_mean=subtractMeanInQuantization,
                                                    modify_in_place=False, bucket_size=bucket_size)[0]

        elif backprop_quantization_style == 'complicated':
            quantizeFunctions = [quantization.uniformQuantization_variable(s, type_of_scaling=type_of_scaling,
                                                    stochastic_rounding=stochasticRounding,
                                                    max_element=maxElementAllowedForQuantization,
                                                    subtract_mean=subtractMeanInQuantization,
                                                    modify_in_place=False, bucket_size=bucket_size) \
                                 for _ in model.parameters()]
        else:
            raise ValueError('The specified backprop_quantization_style not recognized')

    fields = train_loader.dataset.fields
    # Collect features.
    src_features = collect_features(train_loader.dataset, fields)
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))

    train_loss = make_loss_compute(model, fields["tgt"].vocab, train_loader.dataset,
                                   options.copy_attn, options.copy_attn_force, use_distillation_loss, teacher_model)
    #for validation we don't use distilled loss; it would screw up the perplexity computation
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, test_loader.dataset,
                                   options.copy_attn, options.copy_attn_force)

    trunc_size = options.truncated_decoder  # Badly named...
    shard_size = options.max_generator_batches

    trainer = thf.MyTrainer(model, train_loader, test_loader,
                            train_loss, valid_loss, optim,
                            trunc_size, shard_size)

    perplexity_epochs = []
    for epoch in range(options.start_epoch, options.epochs + 1):
        train_stats = onmt.Statistics()
        model.train()
        for idx_batch, batch in enumerate(train_loader):

            model.zero_grad()

            if quantizeWeights:
                if step_since_last_grad_quant_estimation >= num_estimate_quant_grad:
                    # we save them because we only want to quantize weights to compute gradients,
                    # but keep using non-quantized weights during the algorithm
                    model_state_dict = model.state_dict()
                    for idx, p in enumerate(model.parameters()):
                        if quantize_first_and_last_layer is False:
                            if idx == 0 or idx == num_param_model - 1:
                                continue
                        if backprop_quantization_style == 'truncated':
                            p.data.clamp_(-1, 1)  # TODO: Is this necessary? Clamping the weights?
                        if backprop_quantization_style in ('none', 'truncated'):
                            p.data = quantizeFunctions(p.data)
                        elif backprop_quantization_style == 'complicated':
                            p.data = quantizeFunctions[idx].forward(p.data)
                        else:
                            raise ValueError

            trainer.forward_and_backward(idx_batch, batch, epoch, train_stats, report_func, use_distillation_loss, teacher_model)

            if quantizeWeights:
                if step_since_last_grad_quant_estimation >= num_estimate_quant_grad:
                    model.load_state_dict(model_state_dict)
                    del model_state_dict  # free memory

                    if backprop_quantization_style in ('truncated', 'complicated'):
                        for idx, p in enumerate(model.parameters()):
                            if quantize_first_and_last_layer is False:
                                if idx == 0 or idx == num_param_model - 1:
                                    continue
                            #Now some sort of backward. For the none style, we don't do anything.
                            #for the truncated style, we just need to truncate the grad weights
                            #as per the paper here: https://arxiv.org/pdf/1609.07061.pdf
                            #Complicated is my derivation, but unsure whether to use it or not
                            if backprop_quantization_style == 'truncated':
                                p.grad.data[p.data.abs() > 1] = 0
                            elif backprop_quantization_style == 'complicated':
                                p.grad.data = quantizeFunctions[idx].backward(p.grad.data)

            #update parameters after every batch
            trainer.optim.step()

            if step_since_last_grad_quant_estimation >= num_estimate_quant_grad:
                step_since_last_grad_quant_estimation = 0

            step_since_last_grad_quant_estimation += 1

        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())
        perplexity_epochs.append(valid_stats.ppl())

        # 3. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

    if quantizeWeights:
        for idx, p in enumerate(model.parameters()):
            if backprop_quantization_style == 'truncated':
                p.data.clamp_(-1, 1)  # TODO: Is this necessary? Clamping the weights?
            if backprop_quantization_style in ('none', 'truncated'):
                p.data = quantizeFunctions(p.data)
            elif backprop_quantization_style == 'complicated':
                p.data = quantizeFunctions[idx].forward(p.data)
                del quantizeFunctions[idx].saved_for_backward
                quantizeFunctions[idx].saved_for_backward = None  # free memory
            else:
                raise ValueError

    informationDict = {}
    informationDict['perplexity'] = perplexity_epochs
    informationDict['numEpochsTrained'] = options.epochs + 1 - options.start_epoch
    return model, informationDict

def optimize_quantization_points(modelToQuantize, train_loader, test_loader, options, optim=None,
                                 numPointsPerTensor=16, assignBitsAutomatically=False,
                                 use_distillation_loss=False, bucket_size=None):

    print('Preparing training - pre processing tensors')

    if options is None: options = onmt.standard_options.stdOptions
    if not isinstance(options, dict):
        options = mhf.convertToDictionary(options)
    options = handle_options(options)
    options = mhf.convertToNamedTuple(options)

    modelToQuantize.eval()
    quantizedModel = copy.deepcopy(modelToQuantize)

    fields = train_loader.dataset.fields
    train_loss = make_loss_compute(quantizedModel, fields["tgt"].vocab, train_loader.dataset,
                                   options.copy_attn, options.copy_attn_force)
    valid_loss = make_loss_compute(quantizedModel, fields["tgt"].vocab, test_loader.dataset,
                                   options.copy_attn, options.copy_attn_force)
    trunc_size = options.truncated_decoder  # Badly named...
    shard_size = options.max_generator_batches

    numTensorsNetwork = sum(1 for _ in quantizedModel.parameters())
    if isinstance(numPointsPerTensor, int):
        numPointsPerTensor = [numPointsPerTensor] * numTensorsNetwork
    if len(numPointsPerTensor) != numTensorsNetwork:
        raise ValueError('numPointsPerTensor must be equal to the number of tensor in the network')

    scalingFunction = quantization.ScalingFunction(type_scaling='linear', max_element=False,
                                                   subtract_mean=False,
                                                   modify_in_place=False,
                                                   bucket_size=bucket_size)

    quantizedModel.zero_grad()
    dummy_optim = create_optimizer(quantizedModel, options) #dummy optim, just to pass to trainer
    if assignBitsAutomatically:
        trainer = thf.MyTrainer(quantizedModel, train_loader, test_loader,
                               train_loss, valid_loss, dummy_optim,
                               trunc_size, shard_size)
        batch = next(iter(train_loader))
        quantizedModel.zero_grad()
        trainer.forward_and_backward(0, batch, 0, onmt.Statistics(), None)
        fisherInformation = []
        for p in quantizedModel.parameters():
            fisherInformation.append(p.grad.data.norm())
        numPointsPerTensor = qhf.assign_bits_automatically(fisherInformation, numPointsPerTensor, input_is_point=True)
        quantizedModel.zero_grad()
        del trainer
        del optim

    # initialize the points using the percentile function so as to make them all usable
    pointsPerTensor = []
    for idx, p in enumerate(quantizedModel.parameters()):
        initial_points = qhf.initialize_quantization_points(p.data, scalingFunction, numPointsPerTensor[idx])
        initial_points = Variable(initial_points, requires_grad=True)
        # do a dummy backprop so that the grad attribute is initialized. We need this because we call
        # the .backward() function manually later on (since pytorch can't assign variables to model
        # parameters)
        initial_points.sum().backward()
        pointsPerTensor.append(initial_points)

    optionsOpt = copy.deepcopy(mhf.convertToDictionary(options))
    optimizer = create_optimizer(pointsPerTensor, mhf.convertToNamedTuple(optionsOpt))
    trainer = thf.MyTrainer(quantizedModel, train_loader, test_loader,
                            train_loss, valid_loss, dummy_optim,
                            trunc_size, shard_size)
    perplexity_epochs = []


    quantizationFunctions = []
    for idx, p in enumerate(modelToQuantize.parameters()):
        #efficient version of nonUniformQuantization
        quant_fun = quantization.nonUniformQuantization_variable(max_element=False, subtract_mean=False,
                                                                 modify_in_place=False, bucket_size=bucket_size,
                                                                 pre_process_tensors=True, tensor=p.data)

        quantizationFunctions.append(quant_fun)

    print('Pre processing done, training started')

    for epoch in range(options.start_epoch, options.epochs + 1):
        train_stats = onmt.Statistics()
        quantizedModel.train()
        for idx_batch, batch in enumerate(train_loader):

            #zero the gradient
            quantizedModel.zero_grad()

            # quantize the weights
            for idx, p_quantized in enumerate(quantizedModel.parameters()):
                #I am using the efficient version of nonUniformQuantization. The tensors (that don't change across
                #iterations) are saved inside the quantization function, and we only need to pass the quantization
                #points
                p_quantized.data = quantizationFunctions[idx].forward(None, pointsPerTensor[idx].data)

            trainer.forward_and_backward(idx_batch, batch, epoch, train_stats, report_func, use_distillation_loss, modelToQuantize)

            # now get the gradient of the pointsPerTensor
            for idx, p in enumerate(quantizedModel.parameters()):
                pointsPerTensor[idx].grad.data = quantizationFunctions[idx].backward(p.grad.data)[1]

            optimizer.step()

            # after optimzer.step() we need to make sure that the points are still sorted
            for points in pointsPerTensor:
                points.data = torch.sort(points.data)[0]

        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())
        perplexity_epochs.append(valid_stats.ppl())

        # 3. Update the learning rate
        optimizer.updateLearningRate(valid_stats.ppl(), epoch)

    informationDict = {}
    informationDict['perplexity'] = perplexity_epochs
    informationDict['numEpochsTrained'] = options.epochs + 1 - options.start_epoch
    return pointsPerTensor, informationDict
