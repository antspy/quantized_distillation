'''
Implements the model and training techniques detailed in the paper:
"Do deep convolutional neural network really need to be deep and convolutional?"
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Functional
import time
import torch.optim as optim
from torch.nn.init import xavier_uniform, calculate_gain
import math
import copy
import numpy as np
import helpers.functions as mhf
import cnn_models.help_fun as cnn_hf
import quantization
import quantization.help_functions
import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.linear_model

USE_CUDA = torch.cuda.is_available()

#These are the paper specification for teacher model and student model
#Teacher model, 5.3 million parameters
teacherModelSpec = {'spec_conv_layers': [(76, 3, 3), (76, 3, 3), (126, 3, 3), (126, 3, 3), (148, 3, 3),
                                        (148, 3, 3), (148, 3, 3), (148, 3, 3)],
                    'spec_max_pooling': [(1,2,2), (3, 2, 2), (7, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (7, 0.35), (8, 0.4), (9, 0.4)],
                    'spec_linear': [1200, 1200], 'width': 32, 'height': 32}

#smaller model to use as baseline, around 1 million parameters
smallerModelSpec = {'spec_conv_layers': [(75, 5, 5), (50, 5, 5), (50, 5, 5), (25, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [500], 'width': 32, 'height': 32}

class ConvolForwardNet(nn.Module):

    ''' Teacher model as described in the paper :
    "Do deep convolutional neural network really need to be deep and convolutional?"'''

    def __init__(self, width, height, spec_conv_layers, spec_max_pooling, spec_linear, spec_dropout_rates, useBatchNorm=False,
                 useAffineTransformInBatchNorm=False):

        '''
        The structure of the network is: a number of convolutional layers, intermittend max-pooling and dropout layers,
        and a number of linear layers. The max-pooling layers are inserted in the positions specified, as do the dropout
        layers.

        :param spec_conv_layers: list of tuples with (numFilters, width, height) (one tuple for each layer);
        :param spec_max_pooling: list of tuples with (posToInsert, width, height) of max-pooling layers
        :param spec_dropout_rates list of tuples with (posToInsert, rate of dropout) (applied after max-pooling)
        :param spec_linear: list with numNeurons for each layer (i.e. [100, 200, 300] creates 3 layers)
        '''


        super(ConvolForwardNet, self).__init__()


        self.width = width
        self.height = height
        self.conv_layers = []
        self.max_pooling_layers = []
        self.dropout_layers = []
        self.linear_layers = []
        self.max_pooling_positions = []
        self.dropout_positions = []
        self.useBatchNorm = useBatchNorm
        self.batchNormalizationLayers = []

        #creating the convolutional layers
        oldNumChannels = 3
        for idx in range(len(spec_conv_layers)):
            currSpecLayer = spec_conv_layers[idx]
            numFilters = currSpecLayer[0]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            #The padding needs to be such that width and height of the image are unchanges after each conv layer
            padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
            newConvLayer = nn.Conv2d(in_channels=oldNumChannels, out_channels=numFilters,
                                                                    kernel_size=kernel_size, padding=padding)
            xavier_uniform(newConvLayer.weight, calculate_gain('conv2d')) #glorot weight initialization
            self.conv_layers.append(newConvLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm2d(numFilters,
                                                            affine=useAffineTransformInBatchNorm))
            oldNumChannels = numFilters

        #creating the max pooling layers
        for idx in range(len(spec_max_pooling)):
            currSpecLayer = spec_max_pooling[idx]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            self.max_pooling_layers.append(nn.MaxPool2d(kernel_size))
            self.max_pooling_positions.append(currSpecLayer[0])

        #creating the dropout layers
        for idx in range(len(spec_dropout_rates)):
            currSpecLayer = spec_dropout_rates[idx]
            rate = currSpecLayer[1]
            currPosition = currSpecLayer[0]
            if currPosition < len(self.conv_layers):
                #we use dropout2d only for the conv_layers, otherwise we use the usual dropout
                self.dropout_layers.append(nn.Dropout2d(rate))
            else:
                self.dropout_layers.append(nn.Dropout(rate))
            self.dropout_positions.append(currPosition)


        #creating the linear layers
        oldInputFeatures = oldNumChannels * width * height // 2**(2*len(self.max_pooling_layers))
        for idx in range(len(spec_linear)):
            currNumFeatures = spec_linear[idx]
            newLinearLayer = nn.Linear(in_features=oldInputFeatures, out_features=currNumFeatures)
            xavier_uniform(newLinearLayer.weight, calculate_gain('linear'))  # glorot weight initialization
            self.linear_layers.append(newLinearLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm1d(currNumFeatures,
                                                                                 affine=useAffineTransformInBatchNorm))
            oldInputFeatures = currNumFeatures

        #final output layer
        self.out_layer = nn.Linear(in_features=oldInputFeatures, out_features=10)
        xavier_uniform(self.out_layer.weight, calculate_gain('linear'))


        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.max_pooling_layers = nn.ModuleList(self.max_pooling_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batchNormalizationLayers = nn.ModuleList(self.batchNormalizationLayers)
        self.num_conv_layers = len(self.conv_layers)
        self.total_num_layers = self.num_conv_layers + len(self.linear_layers)

    def forward(self, input):

        for idx in range(self.total_num_layers):
            if idx < self.num_conv_layers:
                input = Functional.relu(self.conv_layers[idx](input))
            else:
                if idx == self.num_conv_layers:
                    #if it is the first layer after the convolutional layers, make it as a vector
                    input = input.view(input.size()[0], -1)
                input = Functional.relu(self.linear_layers[idx-self.num_conv_layers](input))

            if self.useBatchNorm:
                input = self.batchNormalizationLayers[idx](input)

            try:
                posMaxLayer = self.max_pooling_positions.index(idx)
                input = self.max_pooling_layers[posMaxLayer](input)
            except ValueError: pass

            try:
                posDropoutLayer = self.dropout_positions.index(idx)
                input = self.dropout_layers[posDropoutLayer](input)
            except ValueError: pass

        input = Functional.relu(self.out_layer(input))

        #No need to take softmax if the loss function is cross entropy
        return input

def train_model(model, train_loader, test_loader, initial_learning_rate = 0.001, use_nesterov=True,
                initial_momentum=0.9, weight_decayL2=0.00022, epochs_to_train=100, print_every=500,
                learning_rate_style='generic', use_distillation_loss=False, teacher_model=None,
                quantizeWeights=False, numBits=8, grad_clipping_threshold=False, start_epoch=0,
                bucket_size=None, quantizationFunctionToUse='uniformLinearScaling',
                backprop_quantization_style='none', estimate_quant_grad_every=1, add_gradient_noise=False,
                ask_teacher_strategy=('always', None), quantize_first_and_last_layer=True,
                mix_with_differentiable_quantization=False):

    # backprop_quantization_style determines how to modify the gradients to take into account the
    # quantization function. Specifically, one can use 'none', where gradients are not modified,
    # 'truncated', where gradient values outside -1 and 1 are truncated to 0 (as per the paper
    # specified in the comments) and 'complicated', which is the temp name for my idea which is slow and complicated
    # to compute

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('To compute distillation loss you have to pass the teacher model')

    if teacher_model is not None:
        teacher_model.eval()

    learning_rate_style = learning_rate_style.lower()
    lr_scheduler = cnn_hf.LearningRateScheduler(initial_learning_rate, learning_rate_style)
    new_learning_rate = initial_learning_rate
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, nesterov=use_nesterov,
                          momentum=initial_momentum, weight_decay=weight_decayL2)
    startTime = time.time()

    pred_accuracy_epochs = []
    percentages_asked_teacher = []
    losses_epochs = []
    informationDict = {}
    last_loss_saved = float('inf')
    step_since_last_grad_quant_estimation = 1
    number_minibatches_per_epoch = len(train_loader)

    if quantizeWeights:
        quantizationFunctionToUse = quantizationFunctionToUse.lower()
        if backprop_quantization_style is None:
            backprop_quantization_style = 'none'
        backprop_quantization_style = backprop_quantization_style.lower()
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
                                                    stochastic_rounding=False,
                                                    max_element=False,
                                                    subtract_mean=False,
                                                    modify_in_place=False, bucket_size=bucket_size)[0]

        elif backprop_quantization_style == 'complicated':
            quantizeFunctions = [quantization.uniformQuantization_variable(s, type_of_scaling=type_of_scaling,
                                                    stochastic_rounding=False,
                                                    max_element=False,
                                                    subtract_mean=False,
                                                    modify_in_place=False, bucket_size=bucket_size) \
                                 for _ in model.parameters()]
        else:
            raise ValueError('The specified backprop_quantization_style not recognized')

        num_parameters = sum(1 for _ in model.parameters())

        def quantize_weights_model(model):
            for idx, p in enumerate(model.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == num_parameters-1:
                        continue #don't quantize first and last layer
                if backprop_quantization_style == 'truncated':
                    p.data.clamp_(-1, 1)
                if backprop_quantization_style in ('none', 'truncated'):
                    p.data = quantizeFunctions(p.data)
                elif backprop_quantization_style == 'complicated':
                    p.data = quantizeFunctions[idx].forward(p.data)
                else:
                    raise ValueError

        def backward_quant_weights_model(model):
            if backprop_quantization_style == 'none':
                return

            for idx, p in enumerate(model.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == num_parameters-1:
                        continue #don't quantize first and last layer

                # Now some sort of backward. For the none style, we don't do anything.
                # for the truncated style, we just need to truncate the grad weights
                # as per the paper here: https://arxiv.org/pdf/1609.07061.pdf
                # if we are quantizing, I put gradient values above 1 to 0.
                # their case it not immediately applicable to ours, but let's try this out
                if backprop_quantization_style == 'truncated':
                    p.grad.data[p.data.abs() > 1] = 0
                elif backprop_quantization_style == 'complicated':
                    p.grad.data = quantizeFunctions[idx].backward(p.grad.data)

    if print_every > number_minibatches_per_epoch:
        print_every = number_minibatches_per_epoch // 2

    try:
        epoch = start_epoch
        for epoch in range(start_epoch, epochs_to_train+start_epoch):
            if mix_with_differentiable_quantization:
                print('=== Starting Quantized Distillation epoch === ')
            model.train()
            print_loss_total = 0
            count_asked_teacher = 0
            count_asked_total = 0
            for idx_minibatch, data in enumerate(train_loader, start=1):

                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        # we save them because we only want to quantize weights to compute gradients,
                        # but keep using non-quantized weights during the algorithm
                        model_state_dict = model.state_dict()
                        quantize_weights_model(model)

                model.zero_grad()
                print_loss, curr_c_teach, curr_c_total = cnn_hf.forward_and_backward(model, data, idx_minibatch, epoch,
                                            use_distillation_loss=use_distillation_loss,
                                            teacher_model=teacher_model,
                                            ask_teacher_strategy=ask_teacher_strategy,
                                            return_more_info=True)
                count_asked_teacher += curr_c_teach
                count_asked_total += curr_c_total

                #load the non-quantize weights and use them for the update. The quantized
                #weights are used only to get the quantized gradient
                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        model.load_state_dict(model_state_dict)
                        del model_state_dict #free memory

                if add_gradient_noise and not quantizeWeights:
                    cnn_hf.add_gradient_noise(model, idx_minibatch, epoch, number_minibatches_per_epoch)

                if grad_clipping_threshold is not False:
                    # gradient clipping
                    for p in model.parameters():
                        p.grad.data.clamp_(-grad_clipping_threshold, grad_clipping_threshold)

                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        backward_quant_weights_model(model)

                optimizer.step()

                if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                    step_since_last_grad_quant_estimation = 0

                step_since_last_grad_quant_estimation += 1

                # print statistics
                print_loss_total += print_loss
                if (idx_minibatch) % print_every == 0:
                    last_loss_saved = print_loss_total / print_every
                    str_to_print = 'Time Elapsed: {}, [Start Epoch: {}, Epoch: {}, Minibatch: {}], loss: {:3f}'.format(
                        mhf.timeSince(startTime), start_epoch+1, epoch + 1, idx_minibatch, last_loss_saved)
                    if pred_accuracy_epochs:
                        str_to_print += ' Last prediction accuracy: {:2f}%'.format(pred_accuracy_epochs[-1]*100)
                    print(str_to_print)
                    print_loss_total = 0

            curr_percentages_asked_teacher = count_asked_teacher/count_asked_total if count_asked_total != 0 else 0
            percentages_asked_teacher.append(curr_percentages_asked_teacher)
            losses_epochs.append(last_loss_saved)
            curr_pred_accuracy = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
            pred_accuracy_epochs.append(curr_pred_accuracy)
            print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy*100))

            if mix_with_differentiable_quantization and epoch != start_epoch + epochs_to_train - 1:
                print('=== Starting Differentiable Quantization epoch === ')
                #the diff quant step is not done at the last epoch, so we end on a quantized distillation epoch
                model_state_dict = optimize_quantization_points(model, train_loader, test_loader, new_learning_rate,
                                            initial_momentum=initial_momentum, epochs_to_train=1, print_every=print_every,
                                            use_nesterov=use_nesterov,
                                            learning_rate_style=learning_rate_style, numPointsPerTensor=2**numBits,
                                            assignBitsAutomatically=True, bucket_size=bucket_size,
                                            use_distillation_loss=True, initialize_method='quantiles',
                                            quantize_first_and_last_layer=quantize_first_and_last_layer)[0]
                model.load_state_dict(model_state_dict)
                del model_state_dict  # free memory
                losses_epochs.append(last_loss_saved)
                curr_pred_accuracy = cnn_hf.evaluateModel(model, test_loader, fastEvaluation=False)
                pred_accuracy_epochs.append(curr_pred_accuracy)
                print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy * 100))


            #updating the learning rate
            new_learning_rate, stop_training = lr_scheduler.update_learning_rate(epoch, 1-curr_pred_accuracy)
            if stop_training is True:
                break
            for p in optimizer.param_groups:
                try:
                    p['lr'] = new_learning_rate
                except:pass

    except Exception as e:
        print('An exception occurred: {}\n. Training has been stopped after {} epochs.'.format(e, epoch))
        informationDict['errorFlag'] = True
        informationDict['numEpochsTrained'] = epoch-start_epoch

        return model, informationDict
    except KeyboardInterrupt:
        print('User stopped training after {} epochs'.format(epoch))
        informationDict['errorFlag'] = False
        informationDict['numEpochsTrained'] = epoch - start_epoch
    else:
        print('Finished Training in {} epochs'.format(epoch+1))
        informationDict['errorFlag'] = False
        informationDict['numEpochsTrained'] = epoch + 1 - start_epoch

    if quantizeWeights:
       quantize_weights_model(model)

    if mix_with_differentiable_quantization:
        informationDict['numEpochsTrained'] *= 2

    informationDict['percentages_asked_teacher'] = percentages_asked_teacher
    informationDict['predictionAccuracy'] = pred_accuracy_epochs
    informationDict['lossSaved'] = losses_epochs
    return model, informationDict

def optimize_quantization_points(modelToQuantize, train_loader, test_loader, initial_learning_rate = 1e-5,
                                 initial_momentum=0.9, epochs_to_train=30, print_every=500, use_nesterov=True,
                                 learning_rate_style='generic', numPointsPerTensor=16,
                                 assignBitsAutomatically=False, bucket_size=None,
                                 use_distillation_loss=True, initialize_method='quantiles',
                                 quantize_first_and_last_layer=True):

    print('Preparing training - pre processing tensors')


    numTensorsNetwork = sum(1 for _ in modelToQuantize.parameters())
    initialize_method = initialize_method.lower()
    if initialize_method not in ('quantiles', 'uniform'):
        raise ValueError('The initialization method must be either quantiles or uniform')

    if isinstance(numPointsPerTensor, int):
        numPointsPerTensor = [numPointsPerTensor] * numTensorsNetwork

    if len(numPointsPerTensor) != numTensorsNetwork:
        raise ValueError('numPointsPerTensor must be equal to the number of tensor in the network')

    if quantize_first_and_last_layer is False:
        numPointsPerTensor = numPointsPerTensor[1:-1]

    #same scaling function that is used inside nonUniformQUantization. It is important they are the same
    scalingFunction = quantization.ScalingFunction('linear', False, False, bucket_size, False)


    #if assigning bits automatically, use the 2-norm of the gradient to determine weights importance
    if assignBitsAutomatically:
        num_to_estimate_grad = 5
        modelToQuantize.zero_grad()
        for idx_minibatch, batch in enumerate(train_loader, start=1):
            cnn_hf.forward_and_backward(modelToQuantize, batch, idx_batch=idx_minibatch, epoch=0,
                                                     use_distillation_loss=False)
            if idx_minibatch >= num_to_estimate_grad:
                break

        #now we compute the 2-norm of the gradient for each parameter
        fisherInformation = []
        for idx, p in enumerate(modelToQuantize.parameters()):
            if quantize_first_and_last_layer is False:
                if idx == 0 or idx == numTensorsNetwork - 1:
                    continue
            fisherInformation.append((p.grad.data/num_to_estimate_grad).norm())

        #zero the grad we computed
        modelToQuantize.zero_grad()

        #now we use a simple linear proportion to assign bits
        #the minimum number of points is half what was given as input
        numPointsPerTensor = quantization.help_functions.assign_bits_automatically(fisherInformation,
                                                                                   numPointsPerTensor,
                                                                                   input_is_point=True)

    #initialize the points using the percentile function so as to make them all usable
    pointsPerTensor = []
    if initialize_method == 'quantiles':
        for idx, p in enumerate(modelToQuantize.parameters()):
            if quantize_first_and_last_layer is True:
                currPointsPerTensor = numPointsPerTensor[idx]
            else:
                if idx == 0 or idx == numTensorsNetwork - 1:
                    continue
                currPointsPerTensor = numPointsPerTensor[idx-1]
            initial_points = quantization.help_functions.initialize_quantization_points(p.data,
                                                                                        scalingFunction,
                                                                                        currPointsPerTensor)
            initial_points = Variable(initial_points, requires_grad=True)
            # do a dummy backprop so that the grad attribute is initialized. We need this because we call
            # the .backward() function manually later on (since pytorch can't assign variables to model
            # parameters)
            initial_points.sum().backward()
            pointsPerTensor.append(initial_points)
    elif initialize_method == 'uniform':
        for numPoint in numPointsPerTensor:
            initial_points = torch.FloatTensor([x/(numPoint-1) for x in range(numPoint)])
            if USE_CUDA: initial_points = initial_points.cuda()
            initial_points = Variable(initial_points, requires_grad=True)
            # do a dummy backprop so that the grad attribute is initialized. We need this because we call
            # the .backward() function manually later on (since pytorch can't assign variables to model
            # parameters)
            initial_points.sum().backward()
            pointsPerTensor.append(initial_points)
    else: raise ValueError

    #dealing with 0 momentum
    options_optimizer = {}
    if initial_momentum != 0: options_optimizer = {'momentum':initial_momentum, 'nesterov':use_nesterov}
    optimizer = optim.SGD(pointsPerTensor, lr=initial_learning_rate, **options_optimizer)

    lr_scheduler = cnn_hf.LearningRateScheduler(initial_learning_rate, learning_rate_style)
    startTime = time.time()

    pred_accuracy_epochs = []
    losses_epochs = []
    last_loss_saved = float('inf')
    number_minibatches_per_epoch = len(train_loader)

    if print_every > number_minibatches_per_epoch:
        print_every = number_minibatches_per_epoch // 2

    modelToQuantize.eval()
    quantizedModel = copy.deepcopy(modelToQuantize)
    epoch = 0

    quantizationFunctions = []
    for idx, p in enumerate(quantizedModel.parameters()):
        if quantize_first_and_last_layer is False:
            if idx == 0 or idx == numTensorsNetwork - 1:
                continue
        #efficient version of nonUniformQuantization
        quant_fun = quantization.nonUniformQuantization_variable(max_element=False, subtract_mean=False,
                                                                 modify_in_place=False, bucket_size=bucket_size,
                                                                 pre_process_tensors=True, tensor=p.data)

        quantizationFunctions.append(quant_fun)

    print('Pre processing done, training started')

    for epoch in range(epochs_to_train):
        quantizedModel.train()
        print_loss_total = 0
        for idx_minibatch, data in enumerate(train_loader, start=1):

            #zero the gradient of the parameters model
            quantizedModel.zero_grad()
            optimizer.zero_grad()

            #quantize the model parameters
            for idx, p_quantized in enumerate(quantizedModel.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == numTensorsNetwork - 1:
                        continue
                    currIdx = idx - 1
                else: currIdx = idx
                #efficient quantization
                p_quantized.data = quantizationFunctions[currIdx].forward(None, pointsPerTensor[currIdx].data)

            print_loss = cnn_hf.forward_and_backward(quantizedModel, data, idx_minibatch, epoch,
                                        use_distillation_loss=use_distillation_loss,
                                        teacher_model=modelToQuantize)

            #now get the gradient of the pointsPerTensor
            for idx, p in enumerate(quantizedModel.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == numTensorsNetwork - 1:
                        continue
                    currIdx = idx - 1
                else: currIdx = idx
                pointsPerTensor[currIdx].grad.data = quantizationFunctions[currIdx].backward(p.grad.data)[1]

            optimizer.step()

            #after optimzer.step() we need to make sure that the points are still sorted. Implementation detail
            for points in pointsPerTensor:
                points.data = torch.sort(points.data)[0]

            # print statistics
            print_loss_total += print_loss
            if (idx_minibatch) % print_every == 0:
                last_loss_saved = print_loss_total / print_every
                str_to_print = 'Time Elapsed: {}, [Epoch: {}, Minibatch: {}], loss: {:3f}'.format(
                    mhf.timeSince(startTime), epoch + 1, idx_minibatch, last_loss_saved)
                if pred_accuracy_epochs:
                    str_to_print += '. Last prediction accuracy: {:2f}%'.format(pred_accuracy_epochs[-1] * 100)
                print(str_to_print)
                print_loss_total = 0

        losses_epochs.append(last_loss_saved)
        curr_pred_accuracy = cnn_hf.evaluateModel(quantizedModel, test_loader, fastEvaluation=False)
        pred_accuracy_epochs.append(curr_pred_accuracy)
        print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy * 100))

        # updating the learning rate
        new_learning_rate, stop_training = lr_scheduler.update_learning_rate(epoch, 1 - curr_pred_accuracy)
        if stop_training is True:
            break
        for p in optimizer.param_groups:
            try:
                p['lr'] = new_learning_rate
            except:
                pass

    print('Finished Training in {} epochs'.format(epoch + 1))
    informationDict = {'predictionAccuracy': pred_accuracy_epochs,
                       'numEpochsTrained': epoch+1,
                       'lossSaved':losses_epochs}

    #IMPORTANT: When there are batch normalization layers, important information is contained
    #also in the running mean and runnin var values of the batch normalization layers. Since these are not
    #parameters, they don't show up in model.parameter() list (and they don't have quantization points
    #associated with it). So if I return just the optimized quantization points, and quantize the model
    #weight with them, I will have inferior performance because the running mean and var of the batch normalization
    #layers won't be saved. To solve this issue I also return the quantized model state dict, that contains
    #not only the parameter of the models but also this statistics for the batch normalization layers

    return quantizedModel.state_dict(), pointsPerTensor, informationDict
