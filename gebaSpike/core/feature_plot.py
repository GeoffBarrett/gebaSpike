from core.feature_functions import CreateFeatures
from core.default_parameters import openGL, gridLines, feature_spike_size
from core.plot_utils import get_spike_colors
import pyqtgraph.opengl as gl
import numpy as np

feature_name_map = {
    # 'PC1': 'WavePCX!1',
    # 'PC2': 'WavePCX!2',
    # 'PC3': 'WavePCX!3',
    # 'PC4': 'WavePCX!4',
    'PC1': 'WavePCX_scikit!1',
    'PC2': 'WavePCX_scikit!2',
    'PC3': 'WavePCX_scikit!3',
    'PC4': 'WavePCX_scikit!4',
    'Energy': 'Energy',
    'Amplitude': 'Amplitude',
    'Peak': 'Peak',
    'Peak Time': 'PeakTime',
    'Trough': 'Trough',
    'Trough Time': 'TroughTime',
}


def load_features(self):
    """
    This functionn will calculate the features that the user has chosen in the Main Window and save it in the
    self.feature_data dictionary

    :param self:
    :return:
    """

    for feature in [self.x_axis_cb.currentText(), self.y_axis_cb.currentText(), self.z_axis_cb.currentText()]:

        if feature == 'None':
            continue

        feature_function = feature_name_map[feature]

        if self.feature_data is None:
            self.feature_data = {}

        if feature not in self.feature_data.keys():
            self.feature_data[feature] = CreateFeatures(self.tetrode_data, featuresToCalculate=[feature_function])


def plot_features(self):
    """
    This function will take the feature list from the Main Window and plot the data onto the self.feature_win
    window.

    :param self: the main window object
    :return: None
    """

    feature_list = [self.x_axis_cb.currentText(), self.y_axis_cb.currentText(), self.z_axis_cb.currentText()]

    feature_list = [feature for feature in feature_list if feature != 'None']

    for channel in np.arange(self.n_channels):
        Zaxis = False
        data = None
        if len(feature_list) > 2:
            # 3d graph
            feature_x = feature_list[0]
            X = self.feature_data[feature_x][:, channel]

            feature_y = feature_list[1]
            Y = self.feature_data[feature_y][:, channel]

            feature_z = feature_list[2]
            Z = self.feature_data[feature_z][:, channel]

            Zaxis = True

            data = np.vstack((X, Y, Z))
        elif len(feature_list) <= 1:
            return
        else:
            # 2d graph
            feature_x = feature_list[0]
            X = self.feature_data[feature_x][:, channel]
            feature_y = feature_list[1]
            Y = self.feature_data[feature_y][:, channel]
            feature_z = ''
            data = np.vstack((X, Y))

        if openGL:
            # --------------- this is if we want to use the open gl 3d plots ---------------------------- #

            # Add the axis lines to the plot to provide users with orientation
            if gridLines:

                startx = np.amin(X)
                starty = np.amin(Y)

                if Zaxis:
                    startz = np.amin(Z)
                else:
                    startz = 0

                if self.xline is None:
                    self.xline = gl.GLLinePlotItem(pos=np.array([[startx, starty, startz], [np.amax(X), starty, startz]]),
                                                   color=(1, 0, 0, 1), width=2, antialias=True)
                    self.glViewWidget.addItem(self.xline)
                else:
                    self.xline.setData(pos=np.array([[startx, starty, startz], [np.amax(X), starty, startz]]),
                                       color=(1, 0, 0, 1), width=2, antialias=True)

                if self.yline is None:
                    self.yline = gl.GLLinePlotItem(pos=np.array([[startx, starty, startz], [startx, np.amax(Y), startz]]),
                                                   color=(0, 1, 0, 1), width=2, antialias=True)
                    self.glViewWidget.addItem(self.yline)
                else:
                    self.yline.setData(pos=np.array([[startx, starty, startz], [startx, np.amax(Y), startz]]),
                                       color=(0, 1, 0, 1), width=2, antialias=True)

                if Zaxis:
                    if self.zline is None:
                        self.zline = gl.GLLinePlotItem(pos=np.array([[startx, starty, startz], [startx, starty, np.amax(Z)]]),
                                                       color=(0, 0, 1, 1), width=2, antialias=True)
                        self.glViewWidget.addItem(self.zline)
                    else:
                        self.zline.setData(pos=np.array([[startx, starty, startz], [startx, starty, np.amax(Z)]]),
                                           color=(0, 0, 1.0, 0.5), width=2, antialias=True)
                        self.zline.show()
                else:
                    if self.zline is not None:
                        self.zline.hide()

            if self.spike_colors is None:
                get_spike_colors(self)

            if self.scatterItem is not None:
                self.scatterItem.setData(pos=data.T, color=self.spike_colors,
                                         size=feature_spike_size)
            else:
                self.scatterItem = gl.GLScatterPlotItem(pos=data.T, color=self.spike_colors,
                                                        size=feature_spike_size)
                self.glViewWidget.addItem(self.scatterItem)
