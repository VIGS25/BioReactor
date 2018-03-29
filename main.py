import sys
from BioReactor.utils import loadDataFrame, getDataVar
from BioReactor.utils import getTrainTestFiles, getData
from BioReactor.utils import preprocessing
from BioReactor.model import traintest
from matplotlib.backends.backend_pdf import PdfPages
import logging

data_dir = sys.path[0] + '\\Aligned + Interpolated Data\\'

# Change to False if you want to use interpolated Data
onlyOffline = True

if onlyOffline:
    output_pdf = sys.path[0] + '\\Results\\BioReactor_Offline' + '.pdf'
    output_log = sys.path[0] + '\\Results\\BioReactor_Offline' + '.log'
else:
    output_pdf = sys.path[0] + '\\Results\\BioReactor_Online' + '.pdf'
    output_log = sys.path[0] + '\\Results\\BioReactor_Online' + '.log'

logger = logging.getLogger('BioReactor')
logging_format = '%(message)s'
handler = logging.FileHandler(output_log, mode='w')
formatter = logging.Formatter(logging_format)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def perfusionAnalyzer(data_dir, output_pdf, logger, var_=['Titer'],
                      onlyOffline=True):

    import os

    dataFiles = os.listdir(data_dir)

    # Loading all dataframes into a dictionary

    dataframes = {}
    for file in dataFiles:
        dataframes[file] = loadDataFrame(filename=data_dir + file,
                                         logger=logger)

    for var in var_:

        logger.info("\n")
        logger.info("Carrying out the analysis for variable {}".format(var))

        var_off, spec_off, var_counts = getDataVar(dataframes=dataframes,
                                                   var=var, onlyOffline=True)
        trainFiles, testFiles = getTrainTestFiles(var_counts=var_counts,
                                                  mode='ordered')

        if onlyOffline:
            train_off, train_target_off = getData(files=trainFiles,
                                                  var=var_off, spec=spec_off)
            norm_train_off = preprocessing(data=train_off)

        else:
            var_on, spec_on, _ = getDataVar(dataframes=dataframes,
                                            var='Titer', onlyOffline=False)
            train_on, train_target_on = getData(files=trainFiles,
                                                var=var_on, spec=spec_on)
            norm_train_on = preprocessing(data=train_on)

        test, test_target = getData(files=testFiles, var=var_off, spec=spec_off)
        norm_test = preprocessing(data=test)

        logger.info("\nThe Train Files are: \n")
        logger.info(str(trainFiles))

        logger.info("\n The test files are: \n")
        logger.info(str(testFiles))

        if onlyOffline:

            traintest(data=norm_train_off, targets=train_target_off,
                      test_data=norm_test,
                      test_targets=test_target,
                      pdf_name=output_pdf, var=var,
                      onlyOffline=onlyOffline, logger=logger)
        else:
            traintest(data=norm_train_on, targets=train_target_on,
                      test_data=norm_test,
                      test_targets=test_target,
                      pdf_name=output_pdf, var=var,
                      onlyOffline=onlyOffline, logger=logger)


if __name__ == "__main__":
    with PdfPages(output_pdf) as pdf:
        perfusionAnalyzer(data_dir=data_dir, var_=['Titer'],
                          logger=logger, output_pdf=pdf,
                          onlyOffline=onlyOffline)
    handler.close()
