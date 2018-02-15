import time
import smtplib
import torch
import pickle
import os
import tarfile
from email.mime.text import MIMEText
from collections import namedtuple
from collections import OrderedDict
import functools
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import math
import quantization.help_functions as qhf


USE_CUDA = torch.cuda.is_available()


def rsetattr(obj, attr, val):
    'recurrent setattr'

    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


sentinel = object()


def rgetattr(obj, attr, default=sentinel):
    'recurrent getattr'

    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def read_email_info_from_file(filepath):

    '''
    read email username and password from a file. The format is simply

    email_account::: emailAccount@account.com
    password::: password_of_email_account

    '''
    email_account_flag = 'email_account::: '
    password_flag = 'password::: '

    with open(filepath, 'r') as p:
        for line in p.readlines():
            line = line.rstrip()
            if line.startswith(email_account_flag):
                email_account = line[len(email_account_flag):]
                if '@' not in email_account:
                    raise ValueError('File badly formatted; missing "@" in email account')
            elif line.startswith('password::: '):
                password = line[len(password_flag):]
            else:
                raise ValueError('File badly formatted; wrong line identificators. '
                'Lines should start with "{}" and "{}"'.format(email_account_flag, password_flag))

    try:
        return email_account, password
    except:
        raise ValueError('File badly formatted, missing email or password information')

def send_email_yandex(username, password, targets, subject, message, verbose=True):
    try:
        smtp_ssl_host = 'smtp.yandex.com'
        smtp_ssl_port = 465
        email_suffix = '@yandex.com'
        if email_suffix in username:
            username = username
        else:
            if '@' in username:
                raise ValueError('This does not appear to be a yandex email account')
            username = username + email_suffix

        if isinstance(targets, str):
            targets = [targets]

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = ', '.join(targets)

        server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
        server.login(username, password)
        server.sendmail(username, targets, msg.as_string())
        server.quit()
        errMsg = ''
        if verbose:
            print('email sent')
        return True, errMsg
    except Exception as e:
        errMsg = 'Unable to send email: {}'.format(e)
        if verbose:
            print(errMsg)
        return False, errMsg


def asMinutesHours(s):
    h = s // 3600
    s -= h*3600
    m = s // 60
    s -= m * 60
    if h == 0:
        if m == 0:
            return '%ds' % s
        else:
            return '%dm %ds' % (m, s)
    else:
        return '%dh %dm %ds' %(h,m,s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '{}'.format(asMinutesHours(s))


def getNumberOfParameters(model):
    res = 0
    for x in model.parameters():
        res = res + x.data.cpu().numpy().size

    return res

def convertToNamedTuple(dictionary):

    '''

    :param dictionary: converts a dictionary to a named tuple (if possible), so that you can access with the .attribute
                       syntax. This is necessary to use openNMT-py code base
    :return: the namedtuple
    '''

    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def convertToDictionary(named_tuple):
    '''
    :param namedTuple: converts a named tuple  into a dictionary
    :return: the dictionary
    '''

    return named_tuple._asdict()

def extractTarFile(tar_path, extract_path=None):
    if extract_path is None:
        extract_path = os.path.splitext(tar_path)[0]
    try:
        os.mkdir(extract_path)
    except:pass

    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extractTarFile(item.name, "./" + item.name[:item.name.rfind('/')])

def countLinesFile(filepath):
    with open(filepath, 'r') as f:
        count = sum(1 for line in f)
    return count

def remove_files_list(list_filepath):
    infoMsg = ''
    for x in list_filepath:
        try:
            os.remove(x)
        except Exception as e:
            infoMsg += repr(e) + '\n'
    return infoMsg

def convert_state_dict_to_data_parallel(state_dict):

    '''
    Converts a state dict that was saved without data parallel to one tha can be loaded
    by a data parallel module
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    return new_state_dict

def convert_state_dict_from_data_parallel(state_dict):

    '''
    Converts a state dict that was saved with data parallel to one tha can be loaded
    by a non-data parallel module
    '''

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.`
        else:
            raise ValueError('The state_dict passed was not saved by a data parallel instance')
        new_state_dict[name] = v
    return new_state_dict

def num_distinct_elements(numpy_array, tol=1e-8):

    '''returns the number of distinct elements, considering elements closer than tol as the same
    numpy_array must be one dimensional!'''

    aux = numpy_array[~(np.triu(np.abs(numpy_array[:, None] - numpy_array) <= tol, 1)).any(0)]
    #maybe this is better: np.unique(numpy_array.round(decimals=5)).size
    return aux.size

def get_size_reduction(effective_number_bits, bucket_size=256, full_precision_bits=32):

    if bucket_size is None:
        return full_precision_bits/effective_number_bits

    f = full_precision_bits
    k = bucket_size
    b = effective_number_bits
    return (k*f)/(k*b+2*f)

def get_size_quantized_model(model, numBits, quantization_functions, bucket_size=256,
                             type_quantization='uniform', quantizeFirstLastLayer=True):

    'Returns size in MB'

    if numBits is None:
        return sum(p.numel() for p in model.parameters()) * 4 / 1000000


    numTensors = sum(1 for _ in model.parameters())
    if quantizeFirstLastLayer is True:
        def get_quantized_params():
            return model.parameters()
        def get_unquantized_params():
            return iter(())
    else:
        def get_quantized_params():
            return  (p for idx, p in enumerate(model.parameters()) if idx not in (0, numTensors - 1))
        def get_unquantized_params():
            return (p for idx, p in enumerate(model.parameters()) if idx in (0, numTensors - 1))

    count_quantized_parameters = sum(p.numel() for p in get_quantized_params())
    count_unquantized_parameters = sum(p.numel() for p in get_unquantized_params())

    #Now get the best huffmann bit length for the quantized parameters
    actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(get_quantized_params(), quantization_functions,
                                                                   type_quantization, s=2**numBits)

    #Now we can compute the size.
    size_mb = 0
    size_mb += count_unquantized_parameters*4 #32 bits / 8 = 4 byte per parameter
    size_mb += actual_bit_huffmman*count_quantized_parameters/8 #For the quantized parameters we use the mean huffman length
    if bucket_size is not None:
        size_mb += count_quantized_parameters/bucket_size*8  #for every bucket size, we have to save 2 parameters.
                                                             #so we multiply the number of buckets by 2*32/8 = 8
    size_mb = size_mb / 1000000 #to bring it in MB
    return size_mb


def get_entropy(probabilities):

    natural_log = torch.log(1/probabilities)
    natural_log[natural_log == float('inf')] = 0 #this puts all inf in the tensor to 0, so they don't matter for the entropy
    log_2 = natural_log / np.log(2)
    entropy = (probabilities * log_2).sum()
    return entropy

def compute_entropy_layer(layer_out, normalize=False):
    prob_out = torch.nn.functional.softmax(Variable(layer_out), dim=1).data
    curr_entropy = [get_entropy(prob_out[idx_b, :]) for idx_b in range(prob_out.size(0))]
    curr_entropy = torch.FloatTensor(curr_entropy).view(-1, 1)
    if USE_CUDA: curr_entropy = curr_entropy.cuda()
    if normalize:
        # Normalize them with the max possible entropy value, so divide by log_2(n)
        N = layer_out.size(1)
        curr_entropy = curr_entropy / math.log2(N)
    return curr_entropy

class DataLoader(object):

    """
    Simple data loader that wraps the one-epoch generator
    """

    #TODO: Shouldn't this inherit from some torch.utils class?

    #TODO: This probably belongs to the dataset package

    def __init__(self, dataLoaderIterator, length_dataset, batch_size, shuffled, **kwargs):
        self.dataLoaderIterator = dataLoaderIterator
        self.batch_size = batch_size
        self.shuffled = shuffled
        self.length_dataset = length_dataset

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __iter__(self):
        return self.dataLoaderIterator()

    def __len__(self):
        #TODO: shouldn't it rather be int(self.length_dataset/batch_size)?
        return self.length_dataset

class EnsembleModel(nn.Module):
    def __init__(self, modules):
        super(EnsembleModel, self).__init__()

        self.modules_list = nn.ModuleList(modules)

    def forward(self, input):

        num_modules = len(self.modules_list)
        output = self.modules_list[0](input)
        for idx in range(1, num_modules):
            output += self.modules_list[idx](input)
        return output / num_modules
