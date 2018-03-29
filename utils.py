import numpy as np
import pandas as pd


def loadDataFrame(filename, logger):
    """ Loads the file into a pandas dataframe

    Args:
    filename - Path to file

    """

    dataframe = pd.read_excel(io=filename, header=0, na_values=['[]', '', 'NaN'])
    logger.info("Finished Loading File: {} ".format(filename.split('\\')[-1]))

    return dataframe


def getSpectra(dataframe, indices):
    """ Returns the files for training and testing

    Inputs
    -----------
    dataframe:      pd.DataFrame object from which we need to get spectra

    indices:        row values for which we need the spectra

    Returns
    -----------
    spec_vals:      pd.DataFrame object containing spectra values for given
                    indices

    """

    colList = dataframe.columns
    spec_inds = [index for index in range(len(colList))
                 if colList[index].startswith('Spectrum_')]
    spec_cols = colList[spec_inds]

    spec_vals = dataframe[spec_cols].iloc[indices]

    return spec_vals


def getDataOffline(dataframe, var='Titer'):

    """ Returns the offline values for given variable and dataframe

    Inputs
    -----------
    dataframe:      pandas.DataFrame object

    var:            target variable for which offline values is to be returned
                    Default: 'Titer'

    Returns
    -----------
    var_values:    pandas.DataFrame object with offline values for given
                    variable var

    spectra:        pandas.DataFrame object with offline values of spectra

    """

    offline_indices = np.where(pd.isnull(dataframe['Run']) == False)[0]
    var_values = dataframe[var].iloc[offline_indices]

    offline_notNaNs = np.where(pd.isnull(var_values) == False)[0]
    var_values = var_values.iloc[offline_notNaNs]

    spectra = getSpectra(dataframe=dataframe, indices=offline_indices[offline_notNaNs])

    return var_values, spectra


def getDataVar(dataframes, var='Titer', onlyOffline=True):

    """ Returns the spectra and variable values for all dataframes"

    Inputs
    -----------
    dataframes:     Dictionary of dataframes, with file name as key

    var:            target variable, default 'Titer'

    onlyOffline:    boolean indicating whether only offline values needed

    Returns
    -----------
    var_values:     pandas.DataFrame object with variable values from all files

    spectra_values: pandas.DataFrame object with spectra values from all files

    var_counts:     Dictionary with counts of nonNaN variable values, with file name
                    as key

    """

    dataFiles = list(dataframes.keys())

    if onlyOffline:

        file = dataFiles[0]

        var_values = {}
        spectra_values = {}

        dataframe = dataframes[file]
        var_values[file], spectra_values[file] = getDataOffline(dataframe=dataframe, var=var)

        counts = {}
        counts[file] = spectra_values[file].shape[0]

        for fileNumber in range(1, len(dataFiles)):
            file = dataFiles[fileNumber]
            dataframe = dataframes[file]
            var_off, spec = getDataOffline(dataframe, var=var)

            counts[file] = spec.shape[0]

            var_values[file] = var_off
            spectra_values[file] = spec

    else:

        file = dataFiles[0]

        var_values = {}
        spectra_values = {}

        dataframe = dataframes[file]
        notNaNs = np.where(pd.isnull(dataframe[var]) == False)[0]

        var_values[file] = dataframe[var].iloc[notNaNs]
        spectra_values[file] = getSpectra(dataframe=dataframe, indices=notNaNs)

        counts = {}
        counts[file] = spectra_values[file].shape[0]

        for fileNumber in range(1, len(dataFiles)):
            file = dataFiles[fileNumber]
            dataframe = dataframes[file]
            notNaNs = np.where(pd.isnull(dataframe[var]) == False)[0]

            var_values[file] = dataframe[var].iloc[notNaNs]
            spectra_values[file] = getSpectra(dataframe=dataframe, indices=notNaNs)
            counts[file] = spectra_values[file].shape[0]

    return var_values, spectra_values, counts


def getTrainTestFiles(var_counts, train_ratio=0.8, mode='ordered'):

    """ Returns the files for training and testing

    Inputs
    -----------
    var_counts:     Dictionary with counts of nonNaN variable values
                    for all files

    train_ratio:    Ratio for train test split

    mode:           'ordered'- Sort in decreasing value of counts till and add
                    to training files till train_ratio is reached

                    'random' - Have a random ordering of the files and add to
                    training files till train_ratio is reached

    Returns
    -----------
    train_files:    List of files to be used for training
    test_files:     List of files to be used for testing

    """

    total_counts = sum(var_counts.values())
    train_counts = np.floor(total_counts * train_ratio)

    if mode == 'ordered':
        var_counts_new = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)

    elif mode == 'random':
        perm = np.random.permutation(len(var_counts.keys()))
        old_key_order = list(var_counts.keys())
        new_key_order = [old_key_order[i] for i in perm]

        var_counts_new = [(key, var_counts[key]) for key in new_key_order]

    train_files = list()

    counts = 0
    index = 0
    while (counts < train_counts):
        counts += var_counts_new[index][1]
        train_files.append(var_counts_new[index][0])
        index += 1

    test_files = [var_counts_new[i][0] for i in range(index, len(var_counts))]

    return train_files, test_files


def getData(files, var, spec, onlyOffline=True):

    """ Returns the data and target values for given list of files

    Inputs
    -----------
    files:          list of files for which data is to be retrieved

    var:            Dictionary of target values for all dataframes

    spec:           Dictionary of spectrum values for all dataframes

    onlyOffline:    Indicates if only Offline values need to be retrieved

    Returns
    -----------
    data:           (ndata, d) numpy.ndarray of final data
    targets:        (ndata, ) numpy array of target variable

    """

    data = spec[files[0]]
    targets = var[files[0]]

    for fileNumber in range(1, len(files)):
        data = data.append(spec[files[fileNumber]])
        targets = targets.append(var[files[fileNumber]])

    data = np.array(data)
    targets = np.array(targets)

    return data, targets


def preprocessing(data):

    """
    Perform all preprocessing on data

    Inputs
    ------
    data:           (data.shape[0], 3326) numpy.ndarray of spectrum
                    data

    Returns
    -------

    preproc_data:   (data.shape[0], 3326) numpy.ndarray of final data


    """
    from scipy import signal
    from sklearn.preprocessing import normalize

    # Savitzky-Golay filter
    preproc_data = signal.savgol_filter(data, 15, 2, axis=1)

    # normalizing data
    preproc_data = normalize(preproc_data, norm='l2')

    return preproc_data
