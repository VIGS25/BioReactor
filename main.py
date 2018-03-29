import numpy as np
import os
import sys
from BioReactor.utils import loadDataFrame, getDataVar
from BioReactor.utils import getTrainTestFiles, getData
from BioReactor.utils import preprocessing
from BioReactor.model import traintest

data_dir = sys.path[0] + '\\Aligned + Interpolated Data\\'


def perfusionAnalyzer(data_dir, var='Titer'):

    dataFiles = os.listdir(data_dir)

    # Loading all dataframes into a dictionary
    dataframes = {}
    for file in dataFiles:
        dataframes[file] = loadDataFrame(filename=data_dir + file)

    # Getting the offline values
    var_off, spec_off, var_counts = getDataVar(dataframes=dataframes, var=var,
                                               onlyOffline=True)

    # Getting the training and testing files
    trainFiles, testFiles = getTrainTestFiles(var_counts=var_counts,
                                              mode='random')

    # Getting the online values
    var_on, spec_on, _ = getDataVar(dataframes=dataframes, var='Titer',
                                    onlyOffline=False)

    # Offline and Online Training Data
    train_off, train_target_off = getData(files=trainFiles, var=var_off,
                                          spec=spec_off)
    train_on, train_target_on = getData(files=trainFiles, var=var_on,
                                        spec=spec_on)

    # Testing Data is only Offline
    test, test_target = getData(files=testFiles, var=var_off, spec=spec_off)

    # Pre-processed training and testing data
    norm_train_off = preprocessing(data=train_off)
    norm_train_on = preprocessing(data=train_on)
    norm_test = preprocessing(data=test)

    print("\nThe Train Files are: \n")
    print(trainFiles)

    print("\n The test files are: \n")
    print(testFiles)

    # OFFLINE CASE TRAINING AND PREDICTIONS

    traintest(data=norm_train_off, targets=train_target_off, test_data=norm_test,
              test_targets=test_target, onlyOffline=True)

    # ONLINE CASE TRAINING AND PREDICTIONS

    traintest(data=norm_train_on, targets=train_target_on, test_data=norm_test,
              test_targets=test_target, onlyOffline=False)


if __name__ == "__main__":
    perfusionAnalyzer(data_dir=data_dir, var='Titer')