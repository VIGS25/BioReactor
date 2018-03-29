import numpy as np
import pandas as pd
from BioReactor.utils import loadDataFrame
import matplotlib.pyplot as plt
import os


class perfusionProcessor(object):

    """Creates an object of class perfusionProcessor

    Args:

        data_dir: str
            Directory where input data is located

        dataFiles: list of str
            Files to be analyzed

        output_dir: str, default None
            Directory for saving results from any analysis

    """

    def __init__(self, data_dir, dataFiles, logger, colnames=None, output_dir=None):

        self.data_dir = data_dir
        self.dataFiles = [file.split('.')[0] for file in dataFiles]
        self.dataFrames = {}

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = '\\'.join(data_dir.split('\\')[:-2]) + '\\Results\\'

        self._load_data(logger)
        self._rename_columns(col_names=colnames)
        self._get_offlineIndices()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _load_data(self, logger):

        """ Loads all files associated with the perfusion experiment"""

        for file in self.dataFiles:

            filename = self.data_dir + file + '.xlsx'
            self.dataFrames[file] = loadDataFrame(filename=filename, logger=logger)

    def _rename_columns(self, col_names=None):

        """Rename columns in names to arguments in col names"""

        if col_names:
            for file in self.dataFiles:
                self.dataFrames[file].rename(columns=col_names, inplace=True)

        else:
            col_names = {
                "Run": "run_id",
                "BID": "batch_n",
                "Amm": "NH3",
                "Viability": "Via",
                "Ext. pH": "ext_pH",
                "%TRY_Aggregation": "%aggr",
                "TRY_Avg.comp": "avg_comp",
                "TRY_celldiameter": "cell_diameter",
                "%DO": "DO",
                "VCD": "Xv",
                "Ext. VCD": "ext_Xv"}

            for file in self.dataFiles:
                self.dataFrames[file].rename(columns=col_names, inplace=True)

    def _get_offlineIndices(self):

        """Gets the offline indices for all files in the pefusion experiment"""

        self.offLineIndices = dict()

        for file in self.dataFiles:
            self.offLineIndices[file] = np.where(
                pd.isnull(self.dataFrames[file]['run_id']) == False
            )[0]

    def get_TiterNonNaNs(self, file, ifOffline=False):

        """ Get all those places where Titer is not NaN

        Args:
            file: str
                - filename

        Returns:
            indices: list
                - rows in the corresponding dataframe where Titer is not NaN

        """
        titer_vals = self.dataFrames[file]['Titer']
        indices = np.where(pd.isnull(titer_vals) == False)[0]

        if not ifOffline:

            return indices

        else:

            offline_indices = self.offLineIndices[file]

            titer_nonNaNOffline = list()
            for index in offline_indices:

                if index in indices:
                    titer_nonNaNOffline.append(index)

            return np.array(titer_nonNaNOffline)

    def _checkTiterLimits(self):

        self.pastTiterLimits = {}

        for file in self.dataFiles:

            titer_nonNans = self.get_TiterNonNaNs(file=file, ifOffline=False)
            titer_vals = self.dataFrames[file]['Titer'].iloc[titer_nonNans]
            self.pastTiterLimits[file] = np.where(titer_vals > 1)[0]

    def get_var(self, file, var=['Titer'], indices=None):

        """For given variable, provides a dictionary with associated
        column values in all files

        Args:

            var: str, default 'Titer'
                - Variable of interest

            indices: list, default None
                - Indices from which values must be retrieved, If None,
                  then retrieves all

        Returns:

            var_dict: dict
                - dictionary with keys as file names and value as retrieved values

        """

        if len(indices):
            return self.dataFrames[file][var].iloc[indices]

        else:
            return self.dataFrames[file][var]

    def checkNaNsInOffline(self, vars=['Titer']):

        """ For variables of interest present in vars, plot the frequency of NaNs
        found in the offline variables"""

        var_interest = {var: [] for var in vars}

        for var in vars:
            files = list()

            for fileNo in np.arange(1, len(self.dataFiles)+1):
                files.append(fileNo)

                file = self.dataFiles[fileNo-1]
                NaNOffline = np.sum(np.isnan(
                    self.dataFrames[file][var].iloc[self.offLineIndices[file]]))
                NaNOffline = NaNOffline/len(self.offLineIndices[file])

                var_interest[var].append(NaNOffline)

            plt.plot(files, var_interest[var], color='r')
            plt.xlabel('File Number')
            plt.ylabel(var + ' Values')

            plotFileName = var + '_NaN_Frequency'
            plt.savefig(self.output_dir + plotFileName)
            plt.close()

    def plotAnimation(self,  all=False):

        def animate(spec, titer, save_file):
            import matplotlib.animation as animation

            plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

            spec_len = 3326
            x_range = range(spec_len)

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

            im1, = ax1.plot(x_range, spec.iloc[0], color='b')
            ax1.set_ylabel('Spectra Values')

            x = [0]
            y = [titer.iloc[0]]
            ax2.plot(x, y, color='r')

            def updatefig(i):
                im1.set_data(x_range, spec.iloc[i])

                x.append(i)
                y.append(titer.iloc[i])

                ax2.plot(x, y, color='r')
                ax2.set_ylabel('Titer Values')

            ani = animation.FuncAnimation(fig, updatefig,
                                          frames=np.arange(1, len(titer)),
                                          interval=200)
            fig.tight_layout()

            ani.save(save_file, writer=writer)

        out_dir = self.output_dir + 'Spectra_Vs_Titer_Evolution\\'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not all:

            for file in self.dataFiles:

                colList = self.dataFrames[file].columns
                spec_inds = [index for index in range(len(colList))
                             if colList[index].startswith('Spectrum_')]
                spectra_cols = colList[spec_inds]

                indices = self.get_TiterNonNaNs(file=file)
                spectra_val = self.get_var(file=file, var=spectra_cols, indices=indices)
                titer_val = self.get_var(file=file, indices=indices)

                save_file = self.output_dir + 'Spectra_Vs_Titer_Evolution\\' + \
                            ''.join(file.split('.')[:-1]) + '.mp4'

                animate(spec=spectra_val, titer=titer_val, save_file=save_file)

        else:

            file = self.dataFiles[0]
            colList = self.dataFrames[file].columns
            spec_inds = [index for index in range(len(colList))
                         if colList[index].startswith('Spectrum_')]
            spectra_cols = colList[spec_inds]

            indices = self.get_TiterNonNaNs(file=file)

            spectra_val = self.get_var(file=file, var=spectra_cols, indices=indices)
            titer_val = self.get_var(file=file, indices=indices)

            save_file = self.output_dir + 'Spectra_Vs_Titer_Evolution\\' + 'all.mp4'

            for index in range(1, len(self.dataFiles)):

                file = self.dataFiles[index]

                colList = self.dataFrames[file].columns
                spec_inds = [index for index in range(len(colList))
                             if colList[index].startswith('Spectrum_')]
                spectra_cols = colList[spec_inds]

                indices = self.get_TiterNonNaNs(file=file)
                spectra_val = spectra_val.append(self.get_var(file=file,
                                                        var=spectra_cols, indices=indices))
                titer_val = titer_val.append(self.get_var(file=file, indices=indices))

            animate(spec=spectra_val, titer=titer_val, save_file=save_file)
