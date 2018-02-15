import torch
import helpers.functions as mhf
import numpy as np

class LoadingTensorsDataset:

    'A simple loading dataset - loads the tensor that are passed in input'

    def __init__(self, path_train_data, path_test_data):

        self.trainData = torch.load(path_train_data)
        self.testData = torch.load(path_test_data)

    def get_train_loader(self, batch_size):
        return self.get_data_loader('train', batch_size, shuffle=True)

    def get_test_loader(self, batch_size):
        return self.get_data_loader('test', batch_size, shuffle=False)

    def get_data_loader(self, type, batch_size, shuffle=False):
        if batch_size <= 0:
            raise ValueError('batch size must be bigger than zero')

        if type == 'train':
            dataset, labels = self.trainData
        elif type == 'test':
            dataset, labels = self.testData
        else: raise ValueError('Invalid value for type')

        total_amount_data = dataset.size(0)

        def loadIter():

            if shuffle:
                allIndices = list(range(total_amount_data)) #TODO: This is stupidily inefficient. Change when have time
                np.random.shuffle(allIndices)

            currIdx = 0
            while True:

                if currIdx + batch_size > total_amount_data:
                    currData = dataset[currIdx:total_amount_data, :]
                    currLabels = labels[currIdx:total_amount_data]
                    yield currData, currLabels
                    break

                currData = dataset[currIdx:currIdx+batch_size, :]
                currLabels = labels[currIdx:currIdx+batch_size]
                yield currData, currLabels

                currIdx += batch_size

        dataLoader = mhf.DataLoader(loadIter, total_amount_data, batch_size, shuffled=shuffle)

        return dataLoader