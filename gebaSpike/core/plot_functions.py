import os
import time
import numpy as np
from core.default_parameters import openGL, gridLines, feature_spike_opacity, feature_spike_size, unitMode
from core.gui_utils import validate_session
from core.Tint_Matlab import find_unit, getspikes
from core.feature_functions import CreateFeatures
from core.plot_utils import CustomViewBox, get_channel_color, MultiLine
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from core.custom_widgets import GLEllipseROI
from pyqtgraph.Qt import QtCore
from functools import partial


feature_name_map = {
    'PC1': 'WavePCX!1',
    'PC2': 'WavePCX!2',
    'PC3': 'WavePCX!3',
    'PC4': 'WavePCX!4',
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

    Zaxis = False

    channel = int(self.channel_cb.currentText()) -1

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

        # self.ROI = GLEllipseROI([110, 10], [30, 20], pen=(3, 9))
        # self.glViewWidget.addItem(self.ROI)
    else:
        # --------------- this is if we want to try matplotlibwidget ---------------------------- #

        """
        I have put this here to try matplotlib, it seems like panning / zooming with the 3d projection
        matplotlib interface is extremely slow, so I will likely not continue to work on this part. I mainly
        decided to try it out because the ROI code has already been developed via PyQtGraph for non openGL graphs.
        """

        if not self.feature_plot_added:
            self.feature_plot = self.feature_win.getFigure().add_subplot(111, projection='3d')  # add a 3D subplot

        self.feature_plot.clear()
        if Zaxis:
            self.feature_plot.scatter(X, Y, Z)
        else:
            self.feature_plot.scatter(X, Y)

        # labeling the axis so we know which feature is which
        self.feature_plot.set_xlabel(feature_x)
        self.feature_plot.set_ylabel(feature_y)
        self.feature_plot.set_zlabel(feature_z)

        # remove the axis labels, we don't really need to know the value
        self.feature_plot.set_xticklabels([])
        self.feature_plot.set_yticklabels([])
        self.feature_plot.set_zticklabels([])

        self.feature_win.draw()


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

        for roi in self.active_ROI:
            roi.hide()

        self.unit_drag_lines[index].show()  # showing the LineSegmentROI

        self.active_ROI = [self.unit_drag_lines[index]]

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

            unit_data = self.unit_data[index]

            crossed_cells = find_spikes_crossed(points, unit_data)

            self.unit_data[index] = np.delete(unit_data, crossed_cells, axis=0)

            replot_unit(self, index)

            self.unit_drag_lines[index].hide()
            self.active_ROI.remove(self.unit_drag_lines[index])


def find_spikes_crossed(points, unit_data, samples_per_spike=50):
    # calculate the line equation so we can get the points on the line
    slope = getSlope(points)
    y0 = getYIntercept(slope, points[0])
    dx = np.diff(points, axis=0)[0, 0]  # change in x
    x = np.arange(dx) + points[0, 0]

    cross_line = slope * x + y0  # equation for the user's line

    unit_data_bool = np.intersect1d(x, np.arange(samples_per_spike))

    crossed_cells = np.unique(np.where(np.diff(np.sign(cross_line - unit_data[:, unit_data_bool.astype(int)])))[0])

    return crossed_cells


def replot_unit(self, index, cell=None):

    if cell is None:
        cell = self.unit_plots[index][1]

    if index in self.plot_lines.keys():
        self.unit_plots[index][0].removeItem(self.plot_lines[index])
        self.unit_plots[index][0].removeItem(self.avg_plot_lines[index])

    self.plot_lines[index] = MultiLine(
        np.tile(np.arange(self.samples_per_spike), (self.unit_data[index].shape[0], 1)),
        self.unit_data[index], pen_color=get_channel_color(cell))

    self.avg_plot_lines[index] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                           np.mean(self.unit_data[index], axis=0).reshape((1, -1)),
                                           pen_color='w', pen_width=2)

    self.unit_plots[index][0].addItem(self.plot_lines[index])
    self.unit_plots[index][0].addItem(self.avg_plot_lines[index])

    self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
    self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
    self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
    self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
    self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions


def plot_units(self):

    unique_cells = np.unique(self.cut_data)
    # the 0'th cell is the dummy cell for Tint so we will remove that

    unique_cells = unique_cells[unique_cells != 0]
    n_cells = len(unique_cells)

    rows, cols = get_grid_dimensions(n_cells)

    row = 0
    col = 0
    for index in np.arange(len(unique_cells)):

        cell = unique_cells[index]

        if unitMode == 'MatplotWidget':
            """
            This is using the matplotlib widget from PyQt, didn't have the greatest performance, leaving it here just
            for legacy code. Will probably remove it at some point.
            """
            self.unit_plots[index] = [self.unit_win.getFigure().add_subplot(int('%d%d%d' % (rows, cols, index+1)))]
            self.unit_plots[index][0].axis('off')
        else:
            self.unit_plots[index] = [self.unit_win.addPlot(row=row,
                                                           col=col,
                                                           viewBox=CustomViewBox(self, self.unit_win))]
            self.vb[index] = self.unit_plots[index][0].vb

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

        if unitMode == 'MatplotWidget':
            """
            This is using the matplotlib widget from PyQt, didn't have the greatest performance, leaving it here just
            for legacy code. Will probably remove it at some point.
            """

            self.unit_plots[index][0].plot(cell_data[self.channel].T, 'b')
            self.unit_plots[index][0].plot(np.mean(cell_data[self.channel], axis=0), 'k-')

        elif unitMode == 'PyQtDefault':
            """
            This is using the generic plot function for PyQt, turns out to be really slow, just keeping it here
            as legacy code.
            """

            # this uses the default plot() function of a graph

            self.unit_plots[index][0].plot(np.mean(cell_data[self.channel], axis=0), pen=get_channel_color(cell))

            self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
            self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
            self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
            self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
            self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
        else:

            self.unit_plots[index].append(cell)

            self.unit_data[index] = cell_data[self.channel]

            self.plot_lines[index] = MultiLine(np.tile(np.arange(self.samples_per_spike), (cell_data[self.channel].shape[0], 1)),
                                               cell_data[self.channel], pen_color=get_channel_color(cell))

            self.avg_plot_lines[index] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                                   np.mean(cell_data[self.channel], axis=0).reshape((1, -1)),
                                                   pen_color='w', pen_width=2)

            self.unit_plots[index][0].addItem(self.plot_lines[index])
            self.unit_plots[index][0].addItem(self.avg_plot_lines[index])

            self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
            self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
            self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
            self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
            self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions

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

            if self.cut_data is None:
                cut_data, _ = find_unit(os.path.dirname(session_filepath), [tetrode_filename])
                self.cut_data = cut_data[0]
                self.cut_data_original = self.cut_data.copy()  # keep a copy of the original to revert if we want.

            load_features(self)

            plot_features(self)

            plot_units(self)

        else:
            # filename does not exist
            self.choice = None
            self.LogError.signal.emit('TetrodeExistError!%s' % tetrode_filename)

            while self.choice is None:
                time.sleep(0.1)

        return
