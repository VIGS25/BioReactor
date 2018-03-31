### Perfusion Data Analysis

#### Requirements
* Requirements (Best run through [Anaconda](https://conda.io/docs/user-guide/install/download.html))
  + Python 3.5+
  + Scikit-Learn
  + Numpy
  + h5py
  + openpyxl
  + xgboost

* .xlsx files containing aligned and unaligned spectra

#### Running

* To run the code, just download the data files to a directory
* Change the following line in ``main.py`` to the directory where the files are loacted  
```
data_dir = sys.path[0] + '\\Aligned + Interpolated Data\\'
```

* Change the following lines in ``main.py`` for saving the results
```
output_pdf = sys.path[0] + '\\Results\\BioReactor_Offline' + '.pdf'
output_log = sys.path[0] + '\\Results\\BioReactor_Offline' + '.log'
output_pdf = sys.path[0] + '\\Results\\BioReactor_Online' + '.pdf'
output_log = sys.path[0] + '\\Results\\BioReactor_Online' + '.log'

```

* To change the variable for the analysis, edit the ``var_`` variable in the following line in ``main.py``
```
perfusionAnalyzer(data_dir=data_dir, var_=['Titer'],
                          logger=logger, output_pdf=pdf,
                          onlyOffline=onlyOffline)
NOTE: var_ must always be a list
```
* To carry out analysis only using offline data, change the following line in ``main.py``
```
onlyOffline = False
```

* The current program
  + Loads the Perfusion data based on the given directory
  + Splits the data into train-test files based on the mode given
  + Filters and normalizes the data
  + Fits a list of regressors using 5 fold cross-validation and logs the Cross Validation error
  + Fits the list of regressors to training data and logs the test error for the best model
  + Saves the prediction plots and cross validation errors in a pdf file

### Results
A summary of results can be seen in the file [BioReactor_Online.pdf](BioReactor_Online.pdf) and [BioReactor_Online.log](BioReactor_Online.log)






