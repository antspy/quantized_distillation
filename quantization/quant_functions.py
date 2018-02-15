import torch
import numpy as np
import numbers
import quantization
import quantization.help_functions as qhf

class ScalingFunction(object):

    '''
    This class is there to hold two functions: the scaling function for a tensor, and its inverse.
    They are budled together in a class because to be able to invert the scaling, we need to remember
    several parameters, and it is a little uncomfortable to do it manually. The class of course remembers
    correctly.
    '''
    # TODO: Make static version of scale and inv_scale that take as arguments all that is necessary,
    # and then the class can just be a small wrapper about calling scale, saving the arguments,
    # and calling inv. So we would have both ways to call the scaling function, directly and through
    # the class.

    def __init__(self, type_scaling, max_element, subtract_mean, bucket_size, modify_in_place=False):

        type_scaling = type_scaling.lower()
        if type_scaling not in ('linear', 'absmax', 'absnorm'):
            raise ValueError('Incorrect parameter: type of scaling must be "linear", ' 
                             '"absMax" or "absNorm"')

        if bucket_size is not None and (bucket_size <= 0 or not isinstance(bucket_size, int)):
            raise ValueError('Bucket size must be an integer and strictly positive. '
            'Pass None if you want to avoid using buckets')

        if max_element is True:
            if not isinstance(max_element, numbers.Number) or isinstance(max_element, bool):
                raise ValueError('maxElementAllowed must be a number')

        self.type_scaling = type_scaling
        self.max_element = max_element
        self.subtract_mean = subtract_mean
        self.bucket_size = bucket_size
        self.modify_in_place = modify_in_place
        self.tol_diff_zero = 1e-10

        #Things we need to invert the tensor. Set to None, will be populated by scale
        self.mean_tensor = None
        self.original_tensor_size = None
        self.original_tensor_length = None
        self.expected_tensor_size = None

        self.alpha = None            #used if linear scaling
        self.beta = None
        self.idx_min_rows = None
        self.idx_max_rows = None

        self.norm_scaling = None     #used if absNorm or absMax
        self.tensor_sign = None

    def scale_down(self, tensor):

        '''
        Scales the tensor using one of the methods. Note that if bucket_size is not None,
        the shape of the tensor will be changed. This change will be inverted by inv_scale
        '''

        if not self.modify_in_place:
            tensor = tensor.clone()

        if self.subtract_mean:
            self.mean_tensor = tensor.mean()
            tensor.sub_(self.mean_tensor)
        else:
            self.mean_tensor = 0

        if self.max_element is not False:
            tensor[tensor > self.max_element] = self.max_element
            tensor[tensor < -self.max_element] = -self.max_element

        self.original_tensor_size = tensor.size()
        self.original_tensor_length = tensor.numel()
        tensor = qhf.create_bucket_tensor(tensor, self.bucket_size, fill_values='last')
        if self.bucket_size is None:
            tensor = tensor.view(-1)
        self.expected_tensor_size = tensor.size()

        # if tensor is bucketed, it has 2 dimension, otherwise it has 1.
        if self.type_scaling == 'linear':
            if self.bucket_size is None:
                min_rows, idx_min_rows = tensor.min(dim=0, keepdim=True)
                max_rows, idx_max_rows = tensor.max(dim=0, keepdim=True)
            else:
                min_rows, idx_min_rows = tensor.min(dim=1, keepdim=True)
                max_rows, idx_max_rows = tensor.max(dim=1, keepdim=True)
            alpha = max_rows - min_rows
            beta = min_rows
            # If alpha is zero for one row, it means the whole row is 0.
            # So we set alpha = 1 there, to avoid nan and inf, and result won't change
            if self.bucket_size is None:
                if alpha[0] < self.tol_diff_zero:
                    alpha[0] = 1
            else:
                alpha[alpha < self.tol_diff_zero] = 1

            self.alpha = alpha
            self.beta = beta
            self.idx_min_rows = idx_min_rows
            self.idx_max_rows = idx_max_rows

            tensor.sub_(self.beta.expand_as(tensor))
            tensor.div_(self.alpha.expand_as(tensor))

        elif self.type_scaling in ('absmax', 'absnorm'):
            self.tensor_sign = torch.sign(tensor)
            tensor.abs_()
            if self.type_scaling == 'absmax':
                norm_to_use = 'max'
            elif self.type_scaling == 'absnorm':
                norm_to_use = 'norm'
            else: raise ValueError

            if self.bucket_size is None:
                norm_scaling = getattr(tensor, norm_to_use)(p=2)
                if norm_scaling < self.tol_diff_zero:
                    norm_scaling = 1
            else:
                norm_scaling = getattr(tensor, norm_to_use)(p=2, dim=1, keepdim=True)
                norm_scaling[norm_scaling < self.tol_diff_zero] = 1

            self.norm_scaling = norm_scaling.view
            tensor.div_(self.norm_scaling.expand_as(tensor))

        return tensor

    def inv_scale_down(self, tensor):

        "inverts the scaling done before. Note that the max_element truncation won't be inverted"

        if not self.modify_in_place:
            tensor = tensor.clone()

        if tensor.size() != self.expected_tensor_size:
            raise ValueError('The tensor passed has not the expected size.')

        if self.type_scaling == 'linear':
            tensor.mul_(self.alpha.expand_as(tensor))
            tensor.add_(self.beta.expand_as(tensor))
        elif self.type_scaling in ('absmax', 'absnorm'):
            tensor.mul_(self.norm_scaling.expand_as(tensor))
            tensor.mul_(self.tensor_sign)

        tensor.add_(self.mean_tensor)
        tensor = tensor.view(-1)[0:self.original_tensor_length]  # remove the filler values
        tensor = tensor.view(self.original_tensor_size)

        return tensor


def uniformQuantization(tensor, s, type_of_scaling='linear', stochastic_rounding=False,
                        max_element=False, subtract_mean=False, bucket_size=None, modify_in_place=False):

    '''
    Quantizes using the random uniform quantization algorithm the tensor passed using s levels.
    '''

    if not modify_in_place:
        tensor = tensor.clone()

    #we always pass True to modify_in_place because we have already cloned it by this point
    scaling_function = ScalingFunction(type_of_scaling, max_element, subtract_mean,
                                       bucket_size, modify_in_place=True)

    tensor = scaling_function.scale_down(tensor)

    #decrease s by one so as to have exactly s quantization points
    s = s - 1

    if stochastic_rounding:
        #What follows is an in-place version of this code:
        # lVector = torch.floor((tensor * s))
        # probabilities = s * tensor - lVector
        # tensor = lVector / s
        probabilities = s*tensor
        tensor.mul_(s)
        tensor.floor_()
        probabilities -= tensor
        tensor.div_(s)

        currRand = torch.rand(tensor.size())
        currRand = currRand.cuda() if tensor.is_cuda else currRand
        tensor.add_((currRand <= probabilities).float()*1/s)
    else:
        tensor.mul_(s)
        tensor.round_()
        tensor.div_(s)

    tensor = scaling_function.inv_scale_down(tensor)
    return tensor, scaling_function

def nonUniformQuantization(tensor, listQuantizationPoints, max_element=False,
                           subtract_mean=False, modify_in_place=False, bucket_size=None,
                           pre_processed_values=False, search_sorted_obj=None, scaling_function=None,
                           tensors_info=None):

    '''

    :param tensor: the tensor to quantize
    :param listQuantizationPoints: the quantization points to quantize it with
    :param max_element: see ScalingFunction doc
    :param subtract_mean: see ScalingFunction doc
    :param modify_in_place: modify the tensor in place or clone it
    :param bucket_size: the bucket size
    :param pre_processed_values: If True, it expects the tensor to be pre-processed already
    :param inverse_idx_sort: The index of pre-processing

    This function is the bottleneck of the differentiable quantization algorithm. One way to speed it up is to
    avoid during the same operations on the tensor every time; in fact, in the differentiable quantiaztion loop,
    the tensors are always the same and only the listQuantiaztionPoints changes. To take advantage of this,
    you can scale the tensor only once, and sort them. Sorting them speeds up the algorithm.
    If you sort them, you need to retain the indices, and pass inverse_idx_sort (the indices the unsort the array).

    In short to pre-process you have to do something like:

    > scaling_function = ScalingFuntion(....)
    > tensor = scaling_function.scale_down(tensor) #scale it down
    > tensor_info = tensor.type(), tensor.is_cuda
    > tensor = tensor.view(-1).cpu().numpy() #we need a 1 dimensional numpy array
    > indices_sort = np.argsort(tensor) #sorting the array
    > tensor = tensor[indices_sort]
    > inv_idx_sort = np.argsort(indices_sort) #getting the indices to unsort the array
    > nonUniformQuantization(tensor, listQuantizationPoints, inv_idx_sort, scaling_function, tensor_info)

    '''
    if pre_processed_values is True and (search_sorted_obj is None or scaling_function is None or tensors_info is None):
        raise ValueError('If values are preprocessed, all pre processed arguments need to be passed')

    if pre_processed_values is False and not (search_sorted_obj is None and \
                                                      scaling_function is None and tensors_info is None):
        raise ValueError('pre processing is False but you are passing some pre processing values. '
                         'This is probably not what you wanted to do, so to avoid bugs an error is raised')

    if isinstance(listQuantizationPoints, list):
        listQuantizationPoints = torch.Tensor(listQuantizationPoints)

    #we need numpy.searchsorted to make this efficient.
    #There is no pytorch equivalent for now, so I need to convert it to a numpy array first
    if not pre_processed_values:
        if not modify_in_place:
            tensor = tensor.clone()

        # we always pass True to modify_in_place because we have already cloned it by this point
        scaling_function = ScalingFunction(type_scaling='linear', max_element=max_element,
                                           subtract_mean=subtract_mean, bucket_size=bucket_size,
                                           modify_in_place=True)

        tensor = scaling_function.scale_down(tensor)
        tensor_type = tensor.type()
        is_tensor_cuda = tensor.is_cuda
        if is_tensor_cuda:
            numpyTensor = tensor.view(-1).cpu().numpy()
        else:
            numpyTensor = tensor.view(-1).numpy()
    else:
        tensor_type, is_tensor_cuda = tensors_info

    listQuantizationPoints = listQuantizationPoints.cpu().numpy()

    if not pre_processed_values:
        #code taken from
        #https://stackoverflow.com/questions/37841654/find-elements-of-array-one-nearest-to-elements-of-array-two/37842324#37842324
        indicesClosest = np.searchsorted(listQuantizationPoints, numpyTensor, side="left").clip(
            max=listQuantizationPoints.size - 1)
        mask = (indicesClosest > 0) & \
               ((indicesClosest == len(listQuantizationPoints)) |
                (np.fabs(numpyTensor - listQuantizationPoints[indicesClosest - 1]) <
                 np.fabs(numpyTensor - listQuantizationPoints[indicesClosest])))
        indicesClosest = indicesClosest - mask
    else:
        indicesClosest = search_sorted_obj.query(listQuantizationPoints)

    #transforming it back to torch tensors
    tensor = listQuantizationPoints[indicesClosest]
    indicesClosest = torch.from_numpy(indicesClosest).long()
    tensor = torch.from_numpy(tensor).type(tensor_type)

    if is_tensor_cuda:
        tensor = tensor.cuda()
        indicesClosest = indicesClosest.cuda()

    tensor = tensor.view(*scaling_function.expected_tensor_size)
    tensor = scaling_function.inv_scale_down(tensor)
    indicesClosest = indicesClosest.view(-1)[0:scaling_function.original_tensor_length] #rescale the indices, too
    indicesClosest = indicesClosest.view(scaling_function.original_tensor_size)
    return tensor, indicesClosest, scaling_function


class uniformQuantization_variable(torch.autograd.Function):
    def __init__(self, s, type_of_scaling='linear', stochastic_rounding=False, max_element=False,
                 subtract_mean=False, modify_in_place=False, bucket_size=None):
        super(uniformQuantization_variable, self).__init__()
        self.s = s
        self.typeOfScaling = type_of_scaling
        self.stochasticRounding = stochastic_rounding
        self.maxElementAllowed = max_element
        self.subtractMean = subtract_mean
        self.modifyInPlace = modify_in_place
        self.bucket_size = bucket_size

        self.saved_for_backward = None

    def forward(self, input):
        self.saved_for_backward = {}
        self.saved_for_backward['input'] = input.clone()
        quantized_tensor = uniformQuantization(input, s=self.s,
                                             type_of_scaling=self.typeOfScaling,
                                             stochastic_rounding=self.stochasticRounding,
                                             max_element=self.maxElementAllowed,
                                             subtract_mean=self.subtractMean,
                                             modify_in_place=self.modifyInPlace,
                                             bucket_size=self.bucket_size)[0]
        return quantized_tensor

    def backward(self, grad_output):

        #TODO: Add explanation or link to explanation
        #A convoluted derivation tells us the following:

        #TODO: Make sure this works with self.bucket_size = None, too!

        if self.typeOfScaling != 'linear':
            raise ValueError('Linear scaling is necessary to backpropagate')

        if self.subtractMean is True:
            raise NotImplementedError('The backprop function assumes subtractMean to be False for now')

        if self.bucket_size is None:
            raise NotImplementedError('Right now the code does not work with bucket_size None.'
                                      ' Not hard to modify though')

        if self.saved_for_backward is None:
            raise ValueError('Need to have called .forward() to be able to call .backward()')

        tensor = self.saved_for_backward['input']
        # I could save the quantized tensor and scaling function from the forward pass, but too memory expensive.
        quantized_tensor_unscaled, scaling_function = uniformQuantization(tensor, s=self.s,
                                             type_of_scaling=self.typeOfScaling,
                                             stochastic_rounding=self.stochasticRounding,
                                             max_element=self.maxElementAllowed,
                                             subtract_mean=self.subtractMean,
                                             modify_in_place=self.modifyInPlace,
                                             bucket_size=self.bucket_size)


        quantized_tensor_unscaled = scaling_function.scale_down(quantized_tensor_unscaled)
        quantized_tensor_unscaled = quantized_tensor_unscaled.view(-1)[0:scaling_function.original_tensor_length]
        quantized_tensor_unscaled = quantized_tensor_unscaled.view(scaling_function.original_tensor_size)
        alpha = scaling_function.alpha
        beta = scaling_function.beta
        total_num_buckets = alpha.size(0)

        bucketed_tensor_sizes = total_num_buckets, self.bucket_size
        alpha = alpha.expand(*bucketed_tensor_sizes).contiguous().view(-1)\
                                            [0:scaling_function.original_tensor_length]
        beta = beta.expand(*bucketed_tensor_sizes).contiguous().view(-1)\
                                            [0:scaling_function.original_tensor_length]
        idx_max_rows = scaling_function.idx_max_rows
        idx_min_rows = scaling_function.idx_min_rows
        #the index are relative to each bucket; so a 4 in the third row is actually a 4 + 2*bucket_size.
        #So I need to adjust these values with this "adder_for_buckets"
        adder_for_buckets = torch.arange(0, self.bucket_size * total_num_buckets, self.bucket_size).long()
        if idx_max_rows.is_cuda:
            adder_for_buckets = adder_for_buckets.cuda()
        idx_max_rows = idx_max_rows + adder_for_buckets
        idx_min_rows = idx_min_rows + adder_for_buckets
        idx_max_rows = idx_max_rows.expand(*bucketed_tensor_sizes).contiguous().view(-1)\
                                            [0:scaling_function.original_tensor_length]
        idx_min_rows = idx_min_rows.expand(*bucketed_tensor_sizes).contiguous().view(-1)\
                                            [0:scaling_function.original_tensor_length]
        one_vector = torch.ones(scaling_function.original_tensor_length)

        tensor = tensor.view(-1)
        grad_output = grad_output.view(-1)

        #create the sparse matrix grad alpha/grad v
        index_sparse_max = torch.LongTensor(2, scaling_function.original_tensor_length)
        index_sparse_min = torch.LongTensor(2, scaling_function.original_tensor_length)
        index_sparse_max[0,:] = torch.arange(0, scaling_function.original_tensor_length)
        index_sparse_min[0, :] = torch.arange(0, scaling_function.original_tensor_length)
        index_sparse_max[1,:] = idx_max_rows
        index_sparse_min[1,:] = idx_min_rows
        grad_sparse_max = torch.sparse.FloatTensor(index_sparse_max,
                                    one_vector,
                                    torch.Size([scaling_function.original_tensor_length]*2))
        grad_sparse_min = torch.sparse.FloatTensor(index_sparse_min,
                                    one_vector,
                                    torch.Size([scaling_function.original_tensor_length]*2))
        grad_alpha = grad_sparse_max - grad_sparse_min

        if tensor.is_cuda:
            grad_alpha = grad_alpha.cuda()

        output = grad_output + \
                 torch.mm(grad_alpha.t(),
                 (grad_output*(quantized_tensor_unscaled-(tensor-beta)/alpha).view(-1)).view(-1,1))

        output = output.view(scaling_function.original_tensor_size)

        del self.saved_for_backward
        self.saved_for_backward = None
        return output

class nonUniformQuantization_variable(torch.autograd.Function):

    def __init__(self, max_element = False, subtract_mean = False,
                 modify_in_place = False, bucket_size=None, pre_process_tensors=False, tensor=None):

        if pre_process_tensors is True and (tensor is None):
            raise ValueError('To pre-process tensors you need to pass the tensor and the scaling function options')

        super(nonUniformQuantization_variable, self).__init__()
        self.maxElementAllowed = max_element
        self.subtractMean = subtract_mean
        self.modifyInPlace = modify_in_place
        self.bucket_size = bucket_size
        self.savedForBackward = None
        self.pre_process_tensors = pre_process_tensors

        #variables used for preprocessing
        self.search_sorted_obj = None
        self.tensors_info = None
        self.scaling_function = None

        if self.pre_process_tensors:
            self.preprocess(tensor)

    def preprocess(self, tensor):
        if not self.modifyInPlace:
            tensor = tensor.clone()
        scaling_function = quantization.ScalingFunction(type_scaling='linear', max_element=self.maxElementAllowed,
                subtract_mean=self.subtractMean, bucket_size=self.bucket_size, modify_in_place=True)
        tensor = scaling_function.scale_down(tensor)
        tensor_type = tensor.type()
        is_tensor_cuda = tensor.is_cuda
        if is_tensor_cuda:
            numpyTensor = tensor.view(-1).cpu().numpy()
        else:
            numpyTensor = tensor.view(-1).numpy()

        self.search_sorted_obj = SearchSorted(numpyTensor.copy())
        self.tensors_info = (tensor_type, is_tensor_cuda)
        self.scaling_function = scaling_function

    def forward(self, inputTensor, listQuantizationPoints):

        if listQuantizationPoints.dim() != 1:
            raise ValueError('listPoints must be a 1-D tensor')

        numPoints = listQuantizationPoints.size()[0]
        if self.pre_process_tensors:
            quantizedTensor, indicesOfQuantization, scaling_function = nonUniformQuantization(
                None, listQuantizationPoints, modify_in_place=self.modifyInPlace,
                max_element=self.maxElementAllowed, subtract_mean=self.subtractMean, bucket_size=self.bucket_size,
                pre_processed_values=True, search_sorted_obj=self.search_sorted_obj,
                scaling_function=self.scaling_function, tensors_info=self.tensors_info)
        else:
            quantizedTensor, indicesOfQuantization, scaling_function = nonUniformQuantization(
                inputTensor, listQuantizationPoints, modify_in_place=self.modifyInPlace,
                max_element=self.maxElementAllowed, subtract_mean=self.subtractMean, bucket_size=self.bucket_size,
                pre_processed_values=False, search_sorted_obj=None, scaling_function=None, tensors_info=None)

        scalingFactor = scaling_function.alpha
        self.savedForBackward = {'indices':indicesOfQuantization, 'numPoints':numPoints, 'scalingFactor':scalingFactor}
        return quantizedTensor

    def backward(self, grad_output):

        grad_inputTensor = grad_output

        #grad_output is delta Loss / delta output.
        #we want deltaLoss / delta listPoints, so we need to do
        #deltaLoss / delta ListPoints = Sum_j deltaLoss/delta output_j * delta output_j/delta ListPoints
        if self.savedForBackward is None:
            raise ValueError('Need savedIndices to be able to call backward()')
        indices = self.savedForBackward['indices']
        numPoints = self.savedForBackward['numPoints']
        scalingFactor = self.savedForBackward['scalingFactor']
        gradPointTensor = torch.zeros(numPoints)
        if quantization.USE_CUDA: gradPointTensor = gradPointTensor.cuda()

        #Remember that delta output_j/delta ListPoints = 0 if the point is quantized to that particular element,
        #otherwise it is equal to the scaling factor. But the scaling factor is different (depending on the
        #bucket). So we modify the gradient by multiplying by the appropriate scaling factor,
        #so that we can group all the indices together in an efficient fashion. The more obvious (but slower)
        #way of doing this would be to check to which bucket does the current element belong to, and then
        #add to the gradient the scaling factor for that bucket multiplied by 1 or 0 depending on the
        #index (1 if it is the index it has been quantized to, 0 otherwise)
        modified_gradient = grad_output.clone()
        modified_gradient = qhf.create_bucket_tensor(modified_gradient, self.bucket_size)
        modified_gradient *= scalingFactor.expand_as(modified_gradient)
        modified_gradient = modified_gradient.view(-1)[0:grad_output.numel()].view(grad_output.size())

        # To avoid this loop, one can do somehting like
        # unqID, idx, IDsums = np.unique(indices, return_counts=True, return_inverse=True)
        # value_sums = np.bincount(idx, modified_gradient.ravel())
        # I don't see an analogous of np.unique in torch, so for now this loop is good enough.
        for idx in range(numPoints):
            gradPointTensor[idx] = torch.masked_select(modified_gradient, indices == idx).sum()

        self.savedIndices = None #reset it to None
        return grad_inputTensor, gradPointTensor


class SearchSorted:
    def __init__(self, tensor, use_k_optimization=True):

        '''
        use_k_optimization requires storing 4x the size of the tensor.
        If use_k_optimization is True, the class will assume that successive calls will be made with similar k.
        When this happens, we can cut the running time significantly by storing additional variables. If it won't be
        called with successive k, set the flag to False, as otherwise would just consume more memory for no
        good reason
        '''

        indices_sort = np.argsort(tensor)
        self.sorted_tensor = tensor[indices_sort]
        self.inv_indices_sort = np.argsort(indices_sort)
        self.use_k_optimization = use_k_optimization

        if use_k_optimization:
            self.indices_sort = indices_sort

        self.previous_indices_results = None
        self.prev_idx_A_k_pair = None

    def query(self, k):

        midpoints = k[:-1] + np.diff(k) / 2
        idx_count = np.searchsorted(self.sorted_tensor, midpoints)
        idx_A_k_pair = []
        count = 0

        old_obj = 0
        for obj in idx_count:
            if obj != old_obj:
                idx_A_k_pair.append((obj, count))
                old_obj = obj
            count += 1

        if not self.use_k_optimization or self.previous_indices_results is None:
            # creates the index matrix in the sorted case
            final_indices = self._create_indices_matrix(idx_A_k_pair, self.sorted_tensor.shape, len(k))
            # and now unsort it to match the original tensor position
            indicesClosest = final_indices[self.inv_indices_sort]
            if self.use_k_optimization:
                self.prev_idx_A_k_pair = idx_A_k_pair
                self.previous_indices_results = indicesClosest
            return indicesClosest

        old_indices_unsorted = self._create_indices_matrix(self.prev_idx_A_k_pair, self.sorted_tensor.shape, len(k))
        new_indices_unsorted = self._create_indices_matrix(idx_A_k_pair, self.sorted_tensor.shape, len(k))
        mask = new_indices_unsorted != old_indices_unsorted

        self.prev_idx_A_k_pair = idx_A_k_pair
        self.previous_indices_results[self.indices_sort[mask]] = new_indices_unsorted[mask]
        indicesClosest = self.previous_indices_results

        return indicesClosest

    @staticmethod
    def _create_indices_matrix(idx_A_k_pair, matrix_shape, len_quant_points):
        old_idx = 0
        final_indices = np.zeros(matrix_shape, dtype=int)
        for idx_A, idx_k in idx_A_k_pair:
            final_indices[old_idx:idx_A] = idx_k
            old_idx = idx_A
        final_indices[old_idx:] = len_quant_points - 1
        return final_indices
