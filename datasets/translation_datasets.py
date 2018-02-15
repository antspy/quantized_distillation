import os
import urllib.request
import shutil
import datasets
import torch
import subprocess
import onmt
import onmt.Trainer
import onmt.IO
import time
import re
import helpers.functions as mhf
import codecs

USE_CUDA = torch.cuda.is_available()

class TranslationDataset(object):

    'Template for translation datasets download and processing with openNMT codebase'

    def __init__(self, dataFolder, src_language='de', tgt_language='en', pin_memory=False):
        self.dataFolder = dataFolder
        self.pin_memory = pin_memory
        self.src_language = src_language
        self.tgt_language = tgt_language

        try:
            os.mkdir(self.dataFolder)
        except:pass

        if not (isinstance(src_language, str) and isinstance(tgt_language, str)):
            raise ValueError('source and target language parameters must be strings')

        self.trainFilesPath = [os.path.join(self.dataFolder, x) for x in ('train.'+src_language, 'train.'+tgt_language)]
        self.testFilesPath = [os.path.join(self.dataFolder, x) for x in ('test.'+src_language, 'test.'+tgt_language)]
        self.processedFilesPath = [os.path.join(self.dataFolder, 'processed' + x) for x in
                                   ('Vocab.pt', 'Train.pt', 'Test.pt')]
        checkProcessedFiles = self.checkProcessedFiles()

        if (not self.checkDataFiles()) and (not checkProcessedFiles):
            self.download_dataset()

        if not checkProcessedFiles:
            self.process_dataset()

        #these are attributes that will be populated by the load_dataset() method
        self.fields = None
        self.trainSet = None
        self.testSet = None

        self.load_dataset()

    def checkProcessedFiles(self):
        return all(os.path.isfile(x) for x in self.processedFilesPath)

    def checkDataFiles(self):
        return all(os.path.isfile(x) for x in self.trainFilesPath + self.testFilesPath)

    def getTrainLoader(self, batch_size, repeat=False, device=-1):

        if device == -1:
            if USE_CUDA: device = 0

        train_loader = onmt.IO.OrderedIterator(dataset=self.trainSet, batch_size=batch_size,
                                               repeat=repeat, device=device)
        return train_loader

    def getTestLoader(self, batch_size, device=-1):

        if device == -1:
            if USE_CUDA: device = 0

        test_loader = onmt.IO.OrderedIterator(dataset=self.testSet, batch_size=batch_size,
                                              train=False, sort=True, device=device)
        return test_loader

    def download_dataset(self):

        '''This method should download the training and the test set, tokenize and clean the files,
        and save them in the filepath specified by self.trainFilesPath, self.testFilesPath
        '''

        raise NotImplementedError

    def process_dataset(self):
        stdProcessOptions = onmt.standard_options.standardPreProcessingOptions
        stdProcessOptions = mhf.convertToNamedTuple(stdProcessOptions)

        print('Preparing Training...')

        with codecs.open(self.trainFilesPath[0], "r", "utf-8") as src_file:
            src_line = src_file.readline().strip().split()
            _, _, nFeatures = onmt.IO.extract_features(src_line)

        fields = onmt.IO.ONMTDataset.get_fields(nFeatures)
        print("Building Training...")
        train = onmt.IO.ONMTDataset(self.trainFilesPath[0], self.trainFilesPath[1], fields, stdProcessOptions)
        print("Building Vocab...")
        onmt.IO.ONMTDataset.build_vocab(train, stdProcessOptions)

        print("Building Test...")
        test = onmt.IO.ONMTDataset(self.testFilesPath[0], self.testFilesPath[1], fields, stdProcessOptions)
        print("Saving train/test/fields")

        # Can't save fields, so remove/reconstruct at training time.
        with open(self.processedFilesPath[0], 'wb') as processed_vocab, \
             open(self.processedFilesPath[1], 'wb') as processed_train, \
             open(self.processedFilesPath[2], 'wb') as processed_test:

            torch.save(onmt.IO.ONMTDataset.save_vocab(fields), processed_vocab)
            train.fields = []
            test.fields = []
            torch.save(train, processed_train)
            torch.save(test, processed_test)

        print('Saving done.')

    def load_dataset(self):

        print('Loading dataset from {}'.format(self.dataFolder))
        startTime = time.time()
        # Load train and test data.
        self.trainSet = torch.load(self.processedFilesPath[1])
        self.testSet = torch.load(self.processedFilesPath[2])

        #Then load the fields
        fields = onmt.IO.ONMTDataset.load_fields(torch.load(self.processedFilesPath[0]))
        self.fields = dict([(k, f) for (k, f) in fields.items() if k in self.trainSet.examples[0].__dict__])
        self.trainSet.fields = self.fields
        self.testSet.fields = self.fields
        print(' * number of training sentences: %d' % len(self.trainSet))
        print('Dataset loaded in {}'.format(mhf.timeSince(startTime)))


class multi30k_DE_EN(TranslationDataset):

    '''
    Dataset of the  WMT16 Multimodal Translation task German-English:
    http://www.statmt.org/wmt16/multimodal-task.html
    '''

    def __init__(self, dataFolder=None, pin_memory=False):

        dataFolder = dataFolder if dataFolder is not None else \
                            os.path.join(datasets.BASE_DATA_FOLDER, 'multi30k_de_en')
        super().__init__(dataFolder, src_language='de', tgt_language='en', pin_memory=pin_memory)

    def download_dataset(self):
        # downloading and extracting files
        trainUrl = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz'
        testUrl = 'https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz'
        tempPathTrain = os.path.join(self.dataFolder, 'tempDownloadTrain')
        tempPathTest = os.path.join(self.dataFolder, 'tempDownloadTest')

        with urllib.request.urlopen(trainUrl) as response, open(tempPathTrain, 'wb') as out_file:
            print('Downloading {} to {}'.format(trainUrl, tempPathTrain))
            shutil.copyfileobj(response, out_file)
        with urllib.request.urlopen(testUrl) as response, open(tempPathTest, 'wb') as out_file:
            print('Downloading {} to {}'.format(testUrl, tempPathTest))
            shutil.copyfileobj(response, out_file)
        print('Download complete')

        print('Extracting archive')
        mhf.extractTarFile(tempPathTrain, self.dataFolder)
        mhf.extractTarFile(tempPathTest, self.dataFolder)
        try:
            os.remove(tempPathTrain)
            os.remove(tempPathTest)
        except Exception as e:
            print(e)
        print('Extracting done')

        # use moses tokenizer on files
        print('Tokenizing the files using moses tokenizer')
        tokenizerPerl = os.path.join(datasets.PATH_PERL_SCRIPTS_FOLDER, 'tokenizer.perl')
        for fileName in ('train', 'test'):
            for lang in ('de', 'en'):
                fullPath = os.path.join(self.dataFolder, fileName + '.' + lang)
                with open(fullPath, 'r') as fileIn, open(fullPath + '.atok', 'w') as fileOut:
                    subprocess.call(['perl', tokenizerPerl, '-a', '-q', '-no-escape', '-threads', '2', '-l', lang],
                                    stdin=fileIn, stdout=fileOut)
                os.remove(fullPath)
                os.rename(fullPath + '.atok', fullPath)
        print('Tokenizing done')

        #finally, remove any blank lines (opentNMT crashes on blank lines)
        for fileName in ('train', 'test'):
            for lang in ('de', 'en'):
                fullPath = os.path.join(self.dataFolder, fileName + '.' + lang)
                with open(fullPath, 'r') as p:
                    text = p.read()
                text = re.sub('\n+', '\n', text)
                with open(fullPath, 'w') as p:
                    p.write(text)


class onmt_integ_dataset(TranslationDataset):
    '''
    This will download an English-German translation model based on the 200k sentence dataset at
    OpenNMT/IntegrationTesting.
    '''

    def __init__(self, dataFolder=None, pin_memory=False):

        dataFolder = dataFolder if dataFolder is not None else \
            os.path.join(datasets.BASE_DATA_FOLDER, 'openNMT_integ_dataset_de_en')
        super().__init__(dataFolder, src_language='de', tgt_language='en', pin_memory=pin_memory)

    def download_dataset(self):
        # downloading already clean files
        trainUrls = ['https://raw.githubusercontent.com/OpenNMT/IntegrationTesting/master/data/de-train-clean.200K.txt',
                     'https://raw.githubusercontent.com/OpenNMT/IntegrationTesting/master/data/en-train-clean.200K.txt']
        testUrls = ['https://raw.githubusercontent.com/OpenNMT/IntegrationTesting/master/data/de-val-clean.10K.txt',
                    'https://raw.githubusercontent.com/OpenNMT/IntegrationTesting/master/data/en-val-clean.10K.txt']

        for idx, urlFile in enumerate(trainUrls):
            with urllib.request.urlopen(urlFile) as response, open(self.trainFilesPath[idx], 'wb') as out_file:
                print('Downloading {} to {}'.format(urlFile, self.trainFilesPath[idx]))
                shutil.copyfileobj(response, out_file)
        for idx, urlFile in enumerate(testUrls):
            with urllib.request.urlopen(urlFile) as response, open(self.testFilesPath[idx], 'wb') as out_file:
                print('Downloading {} to {}'.format(urlFile, self.testFilesPath[idx]))
                shutil.copyfileobj(response, out_file)
        print('Download complete')

        #TODO: Should i remove blank lines from here? I think it's already cleaned, right?


class WMT13_DE_EN(TranslationDataset):

    'Dataset of the WMT 2013 Europarl v7 DE-EN translation'

    def __init__(self, dataFolder=None, pin_memory=False):

        dataFolder = dataFolder if dataFolder is not None else \
            os.path.join(datasets.BASE_DATA_FOLDER, 'WMT13_DE_EN')
        super().__init__(dataFolder, src_language='de', tgt_language='en', pin_memory=pin_memory)

    def download_dataset(self):

        # downloading and extracting files
        euroParlv7Link = 'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz'
        tempTgzPath = os.path.join(self.dataFolder, 'tempDownload')
        print('Downloading {}. The file is around 600Mb, this may take a while'.format(euroParlv7Link))
        with urllib.request.urlopen(euroParlv7Link) as response, open(tempTgzPath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print('Download complete')

        print('Extracting archive')
        mhf.extractTarFile(tempTgzPath, self.dataFolder)
        # move files in the self.dataFolder instead of the trainingFolder
        for file_name in os.listdir(os.path.join(self.dataFolder, 'training')):
            # now keep only the DE-EN files
            file_path = os.path.join(os.path.join(self.dataFolder, 'training'), file_name)
            if file_name in ('europarl-v7.de-en.de', 'europarl-v7.de-en.en'):
                shutil.move(file_path, os.path.join(self.dataFolder, file_name))
            else:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(e)
        try:
            os.rmdir(os.path.join(self.dataFolder, 'training'))
            os.remove(tempTgzPath)
        except Exception as e:
            print(e)
        print('Extracting done')

        engFile = os.path.join(self.dataFolder, 'europarl-v7.de-en.en')
        deFile = os.path.join(self.dataFolder, 'europarl-v7.de-en.de')

        # use moses tokenizer on files
        print('Tokenizing the files using moses tokenizer')
        tokenizerPerl = os.path.join(datasets.PATH_PERL_SCRIPTS_FOLDER, 'tokenizer.perl')
        with open(deFile, 'r') as deFileIn, open(deFile + '.atok', 'w') as deFileOut:
            subprocess.call(['perl', tokenizerPerl, '-a', '-q', '-no-escape', '-threads', '2', '-l', 'de'],
                            stdin=deFileIn, stdout=deFileOut)
        with open(engFile, 'r') as engFileIn, open(engFile + '.atok', 'w') as engFileOut:
            subprocess.call(['perl', tokenizerPerl, '-a', '-q', '-no-escape', '-threads', '2', '-l', 'en'],
                            stdin=engFileIn, stdout=engFileOut)
        try:
            os.remove(engFile)
            os.remove(deFile)
        except Exception as e:
            print(e)
        print('Tokenizing done')

        engFile += '.atok'
        deFile += '.atok'

        # split the dataset into two pieces, train and test set.

        print('Building training and test set')

        totalNumLines = mhf.countLinesFile(engFile)
        if totalNumLines != mhf.countLinesFile(deFile):
            raise ValueError('The english-german files do not have the same number of lines')
        numTrain = totalNumLines * 9 // 10  # 90% is train, 10% is test
        with open(engFile, 'r') as sourceEng, open(deFile, 'r') as sourceDe, \
                open(self.trainFilesPath[0], 'w') as destTrainDe, open(self.trainFilesPath[1], 'w') as destTrainEng, \
                open(self.testFilesPath[0], 'w') as destTestDe, open(self.testFilesPath[1], 'w') as destTestEng:
            for count, (lineDe, lineEng) in enumerate(zip(sourceDe, sourceEng)):
                #this removes blank lines
                if lineDe == '\n'*len(lineDe): continue
                if count <= numTrain:
                    destTrainDe.write(lineDe)
                    destTrainEng.write(lineEng)
                else:
                    destTestDe.write(lineDe)
                    destTestEng.write(lineEng)
        try:
            os.remove(engFile)
            os.remove(deFile)
        except Exception as e:
            print(e)

        print('Training and test set built')