import torch
import math
import numpy as np
from heapq import heappush, heappop, heapify
import quantization
from collections import defaultdict

def invert_pytorch_vector(pytorch_vector):

    'inverts the vector given as input'

    inv_idx = torch.arange(pytorch_vector.size(0) - 1, -1, -1).long()
    inv_idx = inv_idx.cuda() if pytorch_vector.is_cuda else inv_idx
    return pytorch_vector.index_select(0, inv_idx)

def findFirstNonZeroIndex(pytorch_vector):

    'returns -1 if all the elements are 0'

    lenVec = pytorch_vector.size()[0]
    for idx in range(lenVec):
        if pytorch_vector[idx] != 0:
            return idx
    return -1


def cart2hyperspherical(cartesianCoordinates):
    lenCart = cartesianCoordinates.size()[0]
    squares = cartesianCoordinates**2
    flip_squares = invert_pytorch_vector(squares)
    flip_coord = invert_pytorch_vector(cartesianCoordinates)[1:]
    cumSumSqrt = torch.sqrt(flip_squares.cumsum(dim=0))
    radius = cumSumSqrt[-1]
    angles = torch.zeros(lenCart-1)
    firstNonZeroIdx = findFirstNonZeroIndex(cumSumSqrt)
    cumSumSqrt = cumSumSqrt[1:]
    if firstNonZeroIdx == -1:
        #in this case the vector is all zeros, so radius and angles are 0
        return radius, angles

    if firstNonZeroIdx == 0:
        angles = torch.acos(flip_coord / cumSumSqrt)
    elif firstNonZeroIdx == lenCart-1:
        angles[firstNonZeroIdx - 1] = 0 if flip_coord[firstNonZeroIdx - 1] > 0 else math.pi
    else:
        angles[firstNonZeroIdx-1] = 0 if flip_coord[firstNonZeroIdx-1] > 0 else math.pi
        angles[firstNonZeroIdx:] = torch.acos(flip_coord[firstNonZeroIdx:] / cumSumSqrt[firstNonZeroIdx:])

    angles = invert_pytorch_vector(angles)
    if cartesianCoordinates[-1] < 0:
        angles[-1] = 2*math.pi - angles[-1]
    return radius, angles

def hypershperical2cart(sphericalCoordinates):
    radius = sphericalCoordinates[0]
    angles = sphericalCoordinates[1]
    oneTensor = torch.ones(1)
    oneTensor = oneTensor.cuda() if angles.is_cuda else oneTensor

    cosAngles = torch.cos(angles)
    sinAnglesProd = torch.sin(angles).cumprod(dim=0) #could this be made faster? Maybe with sqrt(1-cosAngles**2)?
    cosAngles = torch.cat((cosAngles, oneTensor))
    sinAnglesProd = torch.cat((oneTensor, sinAnglesProd))

    return radius*sinAnglesProd*cosAngles

def create_bucket_tensor(tensor, bucket_size, fill_values='last'):

    if bucket_size is None:
        return tensor

    tensor = tensor.view(-1)
    if fill_values == 'nan':
        fill_values = float('nan')
    if fill_values == 'last':
        fill_values = tensor[-1]

    total_length = tensor.numel()
    multiple, rest = divmod(total_length, bucket_size)
    if multiple != 0 and rest != 0:
        #if multiple is 0, the num of elements is smaller than bucket size so we operate directly
        #on the tensor passed
        values_to_add = torch.ones(bucket_size-rest)*fill_values
        values_to_add = values_to_add.cuda() if tensor.is_cuda else values_to_add
        # add the fill_values to make the tensor a multiple of the bucket size.
        tensor = torch.cat([tensor, values_to_add])
    if multiple == 0:
        #in this case the tensor is smaller than the bucket size. For consistency we still return it in the same
        #format (i.e. a row) but the number of elements is smaller (and equal to the lenght of the tensor)
        tensor = tensor.view(1, total_length)
    else:
        #this is the bucket tensor. A view of the original tensor suffice
        tensor = tensor.view(-1, bucket_size)
    return tensor


def assign_bits_automatically(gradient_norms, inital_bits_to_assign, input_is_point=False):

    '''
    Given the (estimated) gradient norm of each parameter in a model, uses a simple heuristic to come up with optimal
    distribution of bits, given an initial state. The state can be either a list of numbers (representing the amount
    of bits to use for every tensor) or just a number (and it is assumed the same number of bits is used for every
    tensor. If input_is_point is True, then initial_bits are treated as actual number of points, not bits. This is
    useful if the number of points you use is not a power of 2, so that 3 is a valid number of points to use
    '''

    if isinstance(inital_bits_to_assign, int):
        inital_bits_to_assign = [inital_bits_to_assign] * len(gradient_norms)

    if len(inital_bits_to_assign) != len(gradient_norms):
        raise ValueError('There should be as many gradients as there are initial points.')

    total_to_assign = sum(inital_bits_to_assign)

    if input_is_point:
        temp_points_or_bits_per_tensor = [x // 2 for x in inital_bits_to_assign]
    else:
        temp_points_or_bits_per_tensor = [x - 1 for x in inital_bits_to_assign]
    rest_to_assign = total_to_assign - sum(temp_points_or_bits_per_tensor)
    sum_gradient_norms = sum(gradient_norms)

    #TODO: I don't think the one below works very well with bits, because it is doing a linear proportion. With bits,
    #it should do a logarithmic proportion? To think about it and modify.

    points_or_bits_per_tensor = [y + round(x / sum_gradient_norms * rest_to_assign) for x, y in
                          zip(gradient_norms, temp_points_or_bits_per_tensor)]
    diffPointsToAssign = sum(points_or_bits_per_tensor) - total_to_assign

    if diffPointsToAssign > 0:
        # remove from max
        indexMax = points_or_bits_per_tensor.index(max(points_or_bits_per_tensor))
        points_or_bits_per_tensor[indexMax] -= diffPointsToAssign  # if diffPointsToAssign is big this could bring issues
    elif diffPointsToAssign < 0:
        # add to min
        indexMin = points_or_bits_per_tensor.index(min(points_or_bits_per_tensor))
        points_or_bits_per_tensor[indexMin] += -diffPointsToAssign  # if diffPointsToAssign is big this could bring issues

    return points_or_bits_per_tensor

def initialize_quantization_points(tensor, scaling_function, num_points):

    '''
    Returns a good starting value for the non-uniform optimization algorithm. In particular, we use the percentile
    function so as to concentrate values where they are needed. We need the scaling function because it depends on how
    the tensor will be brought to [0, 1]; we need to use the same function for this. Requires a pytorch tensor as input
    '''

    numpy_param = scaling_function.scale_down(tensor).view(-1) \
                                    [0:scaling_function.original_tensor_length].cpu().numpy()
    initial_points = np.percentile(numpy_param, np.linspace(0, 100, num=num_points))
    initial_points = torch.from_numpy(initial_points).type_as(tensor)
    if tensor.is_cuda: initial_points = initial_points.cuda()

    return initial_points


def huffman_encode(symb2freq):

    """Huffman encode the given dict mapping symbols to weights"""
    #code taken from https://rosettacode.org/wiki/Huffman_coding#Python

    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def get_huffman_encoding_mean_bit_length(model_param_iter, quantization_functions, type_quantization='uniform',
                                         s=None):

    '''
    'returns the mean size of the bit requires to encode everything using huffman encoding'
    :param model_param_iter: the iterator returning model parameters
    :param quantization_functions: the quantization function to use. Either a single one or a list with as many functions
                                   as there are tensors in the model
    :param type_quantization:      Uniform or nonUniform. If nonUniform, the model_param_iter must the the original weights,
                                   not the quantized ones! If uniform, it doesn't matter.
    :return: the mean bit size of encoding the model tensors using huffman encoding
    '''

    type_quantization = type_quantization.lower()
    if type_quantization not in ('uniform', 'nonuniform'):
        raise ValueError('type_quantization not recognized')

    if s is None and type_quantization == 'uniform':
        raise ValueError('If type of quantization is uniform, you must provide s')

    if not isinstance(quantization_functions, list):
        quantization_functions = [quantization_functions]

    single_quant_fun = len(quantization_functions) == 1
    total_length = 0
    frequency = defaultdict(int)
    tol = 1e-5
    for idx, param in enumerate(model_param_iter):
        param = param.clone()
        if hasattr(param, 'data'):
            param = param.data

        total_length += param.numel()
        if single_quant_fun:
            quant_fun = quantization_functions[0]
        else:
            quant_fun = quantization_functions[idx]

        if type_quantization == 'uniform':
            quant_points = [x / (s-1) for x in range(s)]
            q_tensor, scal = quant_fun(param)
            numpy_array = scal.scale_down(q_tensor).view(-1)[0:scal.original_tensor_length].cpu().numpy()
            bin_around_points = [x - tol for x in quant_points]
            bin_indices = np.digitize(numpy_array, bin_around_points).flatten() - 1
        elif type_quantization == 'nonuniform':
            _, bin_indices, _ = quant_fun(param)
            bin_indices = bin_indices.view(-1).cpu().numpy()

        unique, counts = np.unique(bin_indices, return_counts=True)
        for val, count in zip(unique, counts):
            frequency[val] += count

    assert total_length == sum(frequency.values())
    frequency = {x: y/total_length for x, y in frequency.items()}
    huffman_code = huffman_encode(frequency)
    mean_bit_length = sum(frequency[x[0]]*len(x[1]) for x in huffman_code)

    return mean_bit_length

def check_right_bits(tensor_iterator, num_quant_points, bucket_size):

    '''numQ_quant_points is the number of quantization points per tensor. If it is a int, it is assumed it is the same
    for all tensors'''

    if isinstance(num_quant_points, int):
        is_int_quant_points = True
    else:
        is_int_quant_points = False

    scaling_function = quantization.ScalingFunction('linear', False, False, bucket_size=bucket_size)
    for idx_tensor, tensor in enumerate(tensor_iterator):
        if hasattr(tensor, 'data'):
            tensor = tensor.data
        #TODO: This does not work, as you're supposed to use the original scaling factors, not the ones
        #you find in the quantized tensor; for example, in the latter there will always be 1, which is not
        #always correct
        tensor = scaling_function.scale_down(tensor)
        distinct_elements = np.unique(tensor.view(-1).cpu().numpy().round(decimals=5))
        num_distinct_elements = len(distinct_elements)
        if is_int_quant_points:
            curr_num_quant_points = num_quant_points
        else:
            curr_num_quant_points = num_quant_points[idx_tensor]

        if num_distinct_elements > curr_num_quant_points + 3:
            return False

    return True



