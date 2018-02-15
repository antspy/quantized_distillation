import os
import datasets
import torch
import pickle
import urllib
import shutil
import numpy as np
import helpers.functions as mhf

class PennTreeBank(object):
    def __init__(self, dataFolder=None):
        self.dataFolder = dataFolder if dataFolder is not None else os.path.join(datasets.BASE_DATA_FOLDER, 'PennTreeBank')
        self.dictionary = Dictionary()
        try:
            os.mkdir(self.dataFolder)
        except:pass

        self.trainFilePath = os.path.join(self.dataFolder, 'train.txt')
        self.testFilePath = os.path.join(self.dataFolder, 'test.txt')
        self.validFilePath = os.path.join(self.dataFolder, 'valid.txt')

        self.trainSetPath = os.path.join(self.dataFolder, 'trainSet')
        self.testSetPath = os.path.join(self.dataFolder, 'testSet')
        self.validSetPath = os.path.join(self.dataFolder, 'validSet')
        self.dictionaryPath = os.path.join(self.dataFolder, 'dictionary')

        checkProcessedFiles = self.checkProcessedFiles()

        if (not self.checkDataFiles()) and (not checkProcessedFiles):
            #download the files from pytorch example folder, but only if the data are not there and not even the
            #processed files
            baseUrl = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/'
            trainUrl = baseUrl + 'train.txt'
            testUrl = baseUrl + 'test.txt'
            validUrl = baseUrl + 'valid.txt'

            for pathToSave, urlDownload in zip([self.trainFilePath, self.testFilePath, self.validFilePath],
                                                                                    [trainUrl, testUrl, validUrl]):
                print('Downloading {} to {}'.format(urlDownload, pathToSave))
                with urllib.request.urlopen(urlDownload) as response, open(pathToSave, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

            print('Files downloaded')
        else:
            print('Files already downloaded')

        if not checkProcessedFiles:
            print('Processing files')

            self.trainSet = self.tokenize(self.trainFilePath)
            self.testSet = self.tokenize(self.testFilePath)
            self.validSet = self.tokenize(self.validFilePath)

            for pathToSave, dataset in zip([self.trainSetPath, self.testSetPath, self.validSetPath],
                                                                    [self.trainSet, self.testSet, self.validSet]):
                with open(pathToSave, 'wb') as f:
                    torch.save(dataset, f)

            with open(self.dictionaryPath, 'wb') as f:
                pickle.dump(self.dictionary, f)

            print('Files processed')
        else:
            with open(self.dictionaryPath, 'rb') as f:
                self.dictionary = pickle.load(f)

            with open(self.trainSetPath, 'rb') as f:
                self.trainSet = torch.load(f)

            with open(self.testSetPath, 'rb') as f:
                self.testSet = torch.load(f)

            with open(self.validSetPath, 'rb') as f:
                self.validSet = torch.load(f)

            print('Files already processed')

    def getTrainLoader(self, batch_size, length_sequence, force_same_size_batch=False):
        return self.getDataLoader('train', batch_size, length_sequence, shuffle=True,
                                  force_same_size_batch=force_same_size_batch)

    def getTestLoader(self, batch_size, length_sequence, force_same_size_batch=False):
        return self.getDataLoader('test', batch_size,length_sequence, shuffle=False,
                                  force_same_size_batch=force_same_size_batch)

    def getValidLoader(self, batch_size, length_sequence, force_same_size_batch=False):
        return self.getDataLoader('valid', batch_size, length_sequence, shuffle=False,
                                  force_same_size_batch=force_same_size_batch)

    def getDataLoader(self, type, batch_size, length_sequence, shuffle=True, force_same_size_batch=False):

        if type == 'train':
            data = self.trainSet
        elif type == 'test':
            data = self.testSet
        elif type == 'valid':
            data = self.validSet
        else:
            raise ValueError('Invalid type. It must be "train", "test" or "valid"')

        if data.ndimension() != 1:
            raise ValueError('Data in input must be a vector')

        length_data = data.size(0)
        total_amount_data = length_data - length_sequence - 1

        def loadIter():

            if shuffle:
                allIndices = list(range(length_data - length_sequence))
                np.random.shuffle(allIndices)

            countNumData = 0
            while True:
                if countNumData + batch_size < total_amount_data:
                    dimCurrBatch = batch_size
                else:
                    if force_same_size_batch is True:
                        break
                    dimCurrBatch = total_amount_data - countNumData

                currBatchData = torch.LongTensor(dimCurrBatch, length_sequence).zero_()
                if currBatchData.type() != data.type():
                    currBatchData.type_as(data)

                currBatchTarget = torch.LongTensor(dimCurrBatch, length_sequence).zero_()
                if currBatchTarget.type() != data.type():
                    currBatchTarget.type_as(data)

                for j in range(countNumData, countNumData + dimCurrBatch):
                    idxToUse = allIndices[j] if shuffle is True else j
                    currBatchData[j-countNumData, :] = data[idxToUse:idxToUse+length_sequence]
                    currBatchTarget[j-countNumData, :] = data[idxToUse+1:idxToUse+length_sequence+1]

                yield currBatchData, currBatchTarget

                countNumData = countNumData + dimCurrBatch
                if countNumData >= total_amount_data:
                    break

        dataLoader = mhf.DataLoader(loadIter, total_amount_data, batch_size, shuffled=shuffle,
                                    length_sequence=length_sequence,
                                    force_same_size_batch=force_same_size_batch)
        return dataLoader

    def checkDataFiles(self):
        return all(os.path.isfile(x) for x in [self.trainFilePath, self.testFilePath,
                                               self.validFilePath])

    def checkProcessedFiles(self):
        return all(os.path.isfile(x) for x in [self.trainSetPath, self.testSetPath,
                                               self.validSetPath, self.dictionaryPath])

    def tokenize(self, path):

        ''' Tokenizes a text file '''

        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Dictionary(object):

    'Helper class for PennTreeBank dataset '

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

