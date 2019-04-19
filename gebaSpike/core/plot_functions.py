import os
import time
import numpy as np
from core.default_parameters import openGL, gridLines, feature_spike_opacity, feature_spike_size, channel_range, max_spike_plots
from core.gui_utils import validate_session
from core.Tint_Matlab import find_unit, getspikes
from core.feature_functions import CreateFeatures
from core.plot_utils import CustomViewBox, get_channel_color, MultiLine
import pyqtgraph.opengl as gl
import pyqtgraph as pg
# from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
# from core.custom_widgets import GLEllipseROI
from pyqtgraph.Qt import QtCore
from functools import partial


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
    for feature in [self.x_axis_cb.currentText(), self.y_axis_cb.currentText(), self.z_axis_cb.currentText()]:

        if feature == 'None':
            continue

        feature_function = feature_name_map[feature]

        if self.feature_data is None:
            self.feature_data = {}

        if feature not in self.feature_data.keys():
            self.feature_data[feature] = CreateFeatures(self.tetrode_data, featuresToCalculate=[feature_function])


def plot_features(self):

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
        break


def get_spike_colors(self):

    n_spikes = self.tetrode_data.shape[1]

    self.spike_colors = np.ones((n_spikes, 4))
    self.spike_colors[:, -1] = feature_spike_opacity

    unique_cells = np.unique(self.cut_data)
    # the 0'th cell is the dummy cell for Tint so we will remove that
    unique_cells = unique_cells[unique_cells != 0]
    for cell in unique_cells:
        cell_color = get_channel_color(cell)
        cell_bool = np.where(self.cut_data == cell)[0]
        self.spike_colors[cell_bool, :-1] = np.asarray(cell_color)/255


def get_grid_dimensions(n_cells, method='auto'):
    """
    This function will automate the rows and columns for grid of unit plots. Right now it will
    attempt to just make the grid as square as possible. However, Tint does 5 per row, so I might
    conform to that as well.

    method='auto' will keep it as square of a shape as possible
    method='5per' will make it 5 cells per row
    """

    if method == 'auto':
        # try to make the shape as square as possible, if there are 9 cells it will do a
        # 3 by 3 formation.

        if np.sqrt(n_cells).is_integer():
            rows = int(np.sqrt(n_cells))
            cols = int(rows)
        else:

            value1 = int(np.ceil(np.sqrt(n_cells)))
            value2 = int(np.floor(np.sqrt(n_cells)))

            if value1 * value2 < n_cells:
                value2 = int(np.ceil(np.sqrt(n_cells)))

            cols, rows = sorted(np.array([value1, value2]))

        # I prefer there being more columns than rows if necessary.
        if rows <= cols:
            return rows, cols
        else:
            return cols, rows
    elif method == '5per':
        # return 5 units per row

        rows = 5
        cols = int(np.ceil(n_cells/rows))

        return rows, cols


def drag(self, index, ev=None):
    # global vb, lr
    if ev.button() == QtCore.Qt.LeftButton:

        if not self.drag_active:
            for roi in self.active_ROI:
                roi.hide()

            self.unit_drag_lines[index].show()  # showing the LineSegmentROI
            self.active_ROI = [self.unit_drag_lines[index]]
            self.drag_active = True

        # defining the start of the selected region
        points = [[self.vb[index].mapToView(ev.buttonDownPos()).x(),
                  self.vb[index].mapToView(ev.buttonDownPos()).y()],
                  [self.vb[index].mapToView(ev.pos()).x(),
                   self.vb[index].mapToView(ev.pos()).y()]]

        self.unit_drag_lines[index].setPoints(points)

        ev.accept()
    else:
        pg.ViewBox.mouseDragEvent(self.vb[index], ev)


def getSlope(points):
    points_diffs = np.diff(points, axis=0).flatten()
    slope = points_diffs[1] / points_diffs[0]

    if points_diffs[0] == 0:
        print('hi')

    return slope


def getYIntercept(slope, point):
    return point[1] - slope * point[0]


def mouse_click_event(self, index, ev=None):

    if ev.button() == QtCore.Qt.RightButton:
        # open menu
        pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.LeftButton:

        # hopefully drag event
        pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.MiddleButton:
        # then we will accept the changes

        if self.unit_drag_lines[index] in self.active_ROI:
            # then we have an active ROI
            # we will get the x,y positions (rounded to the nearest int) of the selected line
            points = np.rint(np.asarray(self.unit_drag_lines[index].getState()['points']))

            # find which channel the user started in
            channel = get_channel_from_y(points[0, 1], channel_range=channel_range, n_channels=self.n_channels)

            unit_data = self.unit_data[index][channel]

            crossed_cells = find_spikes_crossed(points, unit_data, samples_per_spike=self.samples_per_spike)

            # remove these spikes from all the channels
            for channel in np.arange(self.n_channels):
                self.unit_data[index][channel] = np.delete(self.unit_data[index][channel], crossed_cells, axis=0)

            # update the bool
            cell = self.unit_plots[index][1]
            cell_indices = self.cell_indices[cell]
            new_cell_indices = np.delete(cell_indices, crossed_cells)
            self.cell_indices[cell] = new_cell_indices

            # append invalid cells to the new cell number
            invalid_cells = cell_indices[crossed_cells]

            reconfigure = False
            if cell in self.cell_indices.keys():
                self.cell_indices[int(self.move_to_channel.text())] = np.sort(np.concatenate((self.cell_indices[cell], invalid_cells)))
            else:
                self.cell_indices[int(self.move_to_channel.text())] = invalid_cells
                reconfigure = True

            if not reconfigure:
                # update plots for the invalid cell and the
                replot_unit(self, index)

                invalid_index = get_index_from_cell(self, cell)

                if invalid_index is not None:
                    replot_unit(self, invalid_index)
            else:
                pass

            self.unit_drag_lines[index].hide()
            self.active_ROI.remove(self.unit_drag_lines[index])
            self.drag_active = False


def get_index_from_cell(self, cell):

    for index, value in self.unit_plots.items():
        if len(value) == 1:
            continue
        else:
            if value[1] == cell:
                return index
    return None


def get_channel_y_edges(channel_range=256, n_channels=4):
    return np.arange(n_channels + 1) * -channel_range


def get_channel_from_y(y_value, channel_range=256, n_channels=4):
    """Get the channel to look for the line crossing in"""
    edges = get_channel_y_edges(channel_range=channel_range, n_channels=n_channels)

    for channel in np.arange(n_channels):
        if edges[channel] >= y_value >= edges[channel+1]:
            return channel
    return None


def find_spikes_crossed(points, unit_data, samples_per_spike=50):
    # calculate the line equation so we can get the points on the line
    slope = getSlope(points)
    y0 = getYIntercept(slope, points[0])

    x_values = np.sort(points[:, 0]).flatten()

    start = x_values[0]; stop = x_values[1] + 1
    if start < 0:
        start = 0

    if stop > samples_per_spike:
        stop = samples_per_spike

    x = np.arange(start, stop)

    cross_line = slope * x + y0  # equation for the user's line

    unit_data_bool = np.intersect1d(x, np.arange(samples_per_spike))

    crossed_cells = np.unique(np.where(np.diff(np.sign(cross_line - unit_data[:, unit_data_bool.astype(int)])))[0])

    return crossed_cells


def replot_unit(self, index, cell=None):

    if cell == 0:
        return

    # get the cell for the color
    if cell is None:
        cell = self.unit_plots[index][1]

    for channel in np.arange(self.n_channels):
        if index in self.plot_lines.keys():
            # self.unit_plots[index] is list where the 0'th index is the plot
            self.unit_plots[index][0].removeItem(self.plot_lines[index][channel])
            self.unit_plots[index][0].removeItem(self.avg_plot_lines[index][channel])

        plot_data = self.unit_data[index][channel]

        if plot_data.shape[0] > max_spike_plots:
            plot_data = plot_data[np.linspace(0, self.unit_data[index][channel].shape[0]-1,
                                              num=max_spike_plots).astype(int), :]

        self.plot_lines[index][channel] = MultiLine(
            np.tile(np.arange(self.samples_per_spike), (plot_data.shape[0], 1)),
            plot_data, pen_color=get_channel_color(cell))

        self.avg_plot_lines[index][channel] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                               np.mean(self.unit_data[index][channel], axis=0).reshape((1, -1)),
                                               pen_color='w', pen_width=2)

        self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
        self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])


def add_graph_limits():
    pass


def plot_units(self):

    unique_cells = np.unique(self.cut_data)
    # the 0'th cell is the dummy cell for Tint so we will remove that

    unique_cells = unique_cells[unique_cells != 0]

    self.cell_indices[0] = np.where(self.cut_data == 0)[0]

    n_cells = len(unique_cells)

    rows, cols = get_grid_dimensions(n_cells)

    row = 0
    col = 0
    for index in np.arange(len(unique_cells)):

        cell = unique_cells[index]

        self.unit_plots[index] = [self.unit_win.addPlot(row=row, col=col,
                                                        viewBox=CustomViewBox(self, self.unit_win))]
        self.vb[index] = self.unit_plots[index][0].vb

        self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
        self.unit_plots[index][0].setYRange(0, -self.n_channels * channel_range, padding=0)
        self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
        self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
        self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
        self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
        self.unit_plots[index][0].enableAutoRange(False, False)
        self.unit_plots[index][0].setDownsampling(mode='peak')

        # self.unit_drag_lines[index] = pg.LineSegmentROI([[0, 0], [30, 30]])
        self.unit_drag_lines[index] = pg.PolyLineROI([[0, 0], [30, 30]])
        self.unit_drag_lines[index].hide()
        self.unit_plots[index][0].addItem(self.unit_drag_lines[index])

        self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
        self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

        cell_bool = np.where(self.cut_data == cell)[0]
        cell_data = self.tetrode_data[:, cell_bool, :]

        if self.samples_per_spike is None:
            self.samples_per_spike = cell_data.shape[2]

        if self.n_channels is None:
            self.n_channels = cell_data.shape[0]

        # appending the cell number to the unit_plots list so we know which cell refers to which plot
        self.unit_plots[index].append(cell)

        channel_max = np.amax(cell_data[0])

        self.cell_indices[cell] = cell_bool

        for channel in np.arange(self.n_channels):
            # we will keep the data and the indices

            # shifting the data so that the next channel resides below the previous
            # also making the 1st channel start at a y value of 0
            plot_data = cell_data[channel] - channel*channel_range - channel_max

            plot_data_avg = np.mean(plot_data, axis=0).reshape((1, -1))

            if index not in self.unit_data.keys():
                self.unit_data[index] = {channel: plot_data}
            else:
                self.unit_data[index][channel] = plot_data

            if plot_data.shape[0] > max_spike_plots:
                plot_data = plot_data[np.linspace(0, plot_data.shape[0]-1, num=max_spike_plots).astype(int), :]

            if index not in self.plot_lines.keys():

                self.plot_lines[index] = {channel: MultiLine(np.tile(np.arange(self.samples_per_spike), (plot_data.shape[0], 1)),
                                                   plot_data, pen_color=get_channel_color(cell))}
            else:
                self.plot_lines[index][channel] = MultiLine(np.tile(np.arange(self.samples_per_spike), (plot_data.shape[0], 1)),
                                       plot_data, pen_color=get_channel_color(cell))

            if index not in self.avg_plot_lines.keys():
                self.avg_plot_lines[index] = {channel: MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                                                 plot_data_avg, pen_color='w', pen_width=2)}
            else:
                self.avg_plot_lines[index][channel] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                                                plot_data_avg, pen_color='w', pen_width=2)

            self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
            self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

        col += 1
        if col >= cols:
            col = 0
            row += 1

    self.unit_rows = rows
    self.unit_cols = cols


def manage_features(self):

    session_filename = self.filename.text()

    session_valid, error_raised = validate_session(self, session_filename)

    if session_valid:
        tetrode = self.tetrode_cb.currentText()

        session_filepath = os.path.splitext(self.filename.text())[0]

        tetrode_filename = '%s.%s' % (session_filepath, tetrode)

        if os.path.exists(tetrode_filename):

            if self.tetrode_data is None:

                ts, ch1, ch2, ch3, ch4, spikeparams = getspikes(tetrode_filename)
                self.tetrode_data = np.vstack((ch1, ch2, ch3, ch4)).reshape((4, -1, ch1.shape[1]))
                if self.spike_times is None:
                    self.spike_times = ts

                self.n_channels = self.tetrode_data.shape[0]
                self.samples_per_spike = self.tetrode_data.shape[2]

            if self.cut_data is None:
                cut_data = find_unit([tetrode_filename])
                self.cut_data = cut_data[0]
                self.cut_data_original = self.cut_data.copy()  # keep a copy of the original to revert if we want.

            load_features(self)

            # plot_features(self)

            plot_units(self)

        else:
            # filename does not exist
            self.choice = None
            self.LogError.signal.emit('TetrodeExistError!%s' % tetrode_filename)

            while self.choice is None:
                time.sleep(0.1)

        return
