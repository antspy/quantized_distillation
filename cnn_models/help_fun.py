import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import time
import numpy as np
import helpers.functions as mhf
import random

USE_CUDA = torch.cuda.is_available()

def time_forward_pass(model, train_loader):

    model.eval()
    start_time = time.time()
    for idx_minibatch, data in enumerate(train_loader):

        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)

    end_time = time.time()
    return end_time - start_time


def evaluateModel(model, testLoader, fastEvaluation=True, maxExampleFastEvaluation=10000, k=1):

    'if fastEvaluation is True, it will only check a subset of *maxExampleFastEvaluation* images of the test set'


    model.eval()
    correctClass = 0
    totalNumExamples = 0

    for idx_minibatch, data in enumerate(testLoader):

        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        _, topk_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
        topk_predictions = topk_predictions.t()
        correct = topk_predictions.eq(labels.view(1, -1).expand_as(topk_predictions))
        correctClass += correct.view(-1).float().sum(0, keepdim=True).data[0]
        totalNumExamples += len(labels)

        if fastEvaluation is True and totalNumExamples > maxExampleFastEvaluation:
            break

    return correctClass / totalNumExamples

def forward_and_backward(model, batch, idx_batch, epoch, criterion=None,
                         use_distillation_loss=False, teacher_model=None,
                         temperature_distillation=2, ask_teacher_strategy='always',
                         return_more_info=False):

    #TODO: return_more_info is just there for backward compatibility. A big refactoring is due here, and there one should
    #remove the return_more_info flag

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        criterion = criterion.cuda()

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('To compute distillation loss you need to pass the teacher model')

    if not isinstance(ask_teacher_strategy, tuple):
        ask_teacher_strategy = (ask_teacher_strategy, )

    inputs, labels = batch
    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    if USE_CUDA:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # forward + backward + optimize
    outputs = model(inputs)

    count_asked_teacher = 0

    if use_distillation_loss:
        #if cutoff_entropy_value_distillation is not None, we use the distillation loss only on the examples
        #whose entropy is higher than the cutoff.

        weight_teacher_loss = 0.7

        if 'entropy' in ask_teacher_strategy[0].lower():
            prob_out = torch.nn.functional.softmax(outputs).data
            entropy = [mhf.get_entropy(prob_out[idx_b, :]) for idx_b in range(prob_out.size(0))]

        if ask_teacher_strategy[0].lower() == 'always':
            mask_distillation_loss = torch.ByteTensor([True]*outputs.size(0))
        elif ask_teacher_strategy[0].lower() == 'cutoff_entropy':
            cutoff_entropy_value_distillation = ask_teacher_strategy[1]
            mask_distillation_loss = torch.ByteTensor([entr > cutoff_entropy_value_distillation for entr in entropy])
        elif ask_teacher_strategy[0].lower() == 'random_entropy':
            max_entropy = math.log2(outputs.size(1)) #max possible entropy that happens with uniform distribution
            mask_distillation_loss = torch.ByteTensor([random.random() < entr/max_entropy for entr in entropy])
        elif ask_teacher_strategy[0].lower() == 'incorrect_labels':
            _, predictions = outputs.max(dim=1)
            mask_distillation_loss = (predictions != labels).data.cpu()
        else:
            raise ValueError('ask_teacher_strategy is incorrectly formatted')

        index_distillation_loss = torch.arange(0, outputs.size(0))[mask_distillation_loss.view(-1, 1)].long()
        inverse_idx_distill_loss = torch.arange(0, outputs.size(0))[1-mask_distillation_loss.view(-1, 1)].long()
        if USE_CUDA:
            index_distillation_loss = index_distillation_loss.cuda()
            inverse_idx_distill_loss = inverse_idx_distill_loss.cuda()

        # this criterion is the distillation criterion according to Hinton's paper:
        # "Distilling the Knowledge in a Neural Network", Hinton et al.

        softmaxFunction, logSoftmaxFunction, KLDivLossFunction  = nn.Softmax(dim=1), nn.LogSoftmax(dim=1), nn.KLDivLoss()
        if USE_CUDA:
            softmaxFunction, logSoftmaxFunction = softmaxFunction.cuda(), logSoftmaxFunction.cuda(),
            KLDivLossFunction = KLDivLossFunction.cuda()

        if index_distillation_loss.size() != torch.Size():
            count_asked_teacher = index_distillation_loss.numel()
            # if index_distillation_loss is not empty
            volatile_inputs = Variable(inputs.data[index_distillation_loss, :], requires_grad=False)
            if USE_CUDA: volatile_inputs = volatile_inputs.cuda()
            outputsTeacher = teacher_model(volatile_inputs).detach()
            loss_masked = weight_teacher_loss * temperature_distillation**2 * KLDivLossFunction(
                    logSoftmaxFunction(outputs[index_distillation_loss, :]/ temperature_distillation),
                    softmaxFunction(outputsTeacher / temperature_distillation))
            loss_masked += (1-weight_teacher_loss) * criterion(outputs[index_distillation_loss, :],
                                                               labels[index_distillation_loss])
        else:
            loss_masked = 0

        if inverse_idx_distill_loss.size() != torch.Size():
            #if inverse_idx_distill is not empty
            loss_normal = criterion(outputs[inverse_idx_distill_loss, :], labels[inverse_idx_distill_loss])
        else:
            loss_normal = 0
        loss = loss_masked + loss_normal
    else:
        loss = criterion(outputs, labels)

    loss.backward()

    if return_more_info:
        count_total = inputs.size(0)
        return loss.data[0], count_asked_teacher, count_total
    else:
        return loss.data[0]

def add_gradient_noise(model, idx_batch, epoch, number_minibatches_per_epoch):
    # Adding random gaussian noise as in the paper "Adding gradient noise improves learning
    # for very deep networks"

    for p in model.parameters():
        gaussianNoise = torch.Tensor(p.grad.size())
        if USE_CUDA: gaussianNoise = gaussianNoise.cuda()
        # here nu = 0.01, gamma = 0.55, t is the minibatch count = epoch*total_length +idx_minibatch
        stdDev = (0.01 / (1 + epoch * number_minibatches_per_epoch + idx_batch) ** 0.55) ** 0.5
        gaussianNoise.normal_(0, std=stdDev)
        p.grad.data.add_(gaussianNoise)

class LearningRateScheduler:
    def __init__(self, initial_learning_rate, learning_rate_type='generic'):

        if learning_rate_type not in ('generic', 'cifar100', 'imagenet', 'quant_points_cifar100'):
            raise ValueError('Wrong learning rate type specified')

        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_type = learning_rate_type
        self.current_learning_rate = initial_learning_rate

        #variables for the generic method
        self.old_validation_error = float('inf')
        self.epochs_since_validation_error_dropped = 0
        self.total_number_of_learning_rate_halves = 0
        self.epochs_to_wait_for_halving = 0

        #variables for quant_points_cifar100
        self.best_validation_error = float('inf')

    def update_learning_rate(self, epoch, validation_error):

        if self.learning_rate_type == 'cifar100':
            optim_factor = 0
            if (epoch > 160):
                optim_factor = 3
            elif (epoch > 120):
                optim_factor = 2
            elif (epoch > 60):
                optim_factor = 1
            new_learning_rate = self.initial_learning_rate * math.pow(0.2, optim_factor)
            self.current_learning_rate = new_learning_rate
            stop_training = False
            return new_learning_rate, stop_training

        if self.learning_rate_type == 'imagenet':
            new_learning_rate = self.initial_learning_rate * (0.1 ** (epoch // 30))
            stop_training = False

            return new_learning_rate, stop_training

        if self.learning_rate_type in ('generic', 'quant_points_cifar100'):
            if self.learning_rate_type == 'generic':
                epoch_to_wait_before_reducing_rate = 10
                epochs_to_wait_after_halving = 8
                epoch_to_wait_before_stopping = 30
                total_halves_before_stopping = 11
            elif self.learning_rate_type == 'quant_points_cifar100':
                epoch_to_wait_before_reducing_rate = 2
                epochs_to_wait_after_halving = 0
                epoch_to_wait_before_stopping = float('inf')
                total_halves_before_stopping = float('inf')
            else: raise ValueError

            new_learning_rate = self.current_learning_rate
            stop_training = False

            # we have a 0.1% error band
            if validation_error + 0.001 < self.old_validation_error:
                self.old_validation_error = validation_error
                self.epochs_since_validation_error_dropped = 0
            else:
                self.epochs_since_validation_error_dropped += 1

            self.epochs_to_wait_for_halving = max(self.epochs_to_wait_for_halving - 1, 0)
            if self.epochs_since_validation_error_dropped >= epoch_to_wait_before_reducing_rate and \
                            self.epochs_to_wait_for_halving == 0:
                # if validation error does not drop for 10 epochs in a row, halve the learning rate
                # but don't halve it for at least 8 epochs after halving.
                self.epochs_to_wait_for_halving = epochs_to_wait_after_halving
                self.total_number_of_learning_rate_halves += 1
                new_learning_rate = self.current_learning_rate / 2
                self.current_learning_rate = new_learning_rate

            if self.epochs_since_validation_error_dropped > epoch_to_wait_before_stopping or \
                            self.total_number_of_learning_rate_halves > total_halves_before_stopping:
                # stop training if validation rate hasn't dropped in 30 epochs or if learning rates was halved 11 times already
                # i.e. it was reduced by 2048 times.
                stop_training = True

            return new_learning_rate, stop_training
