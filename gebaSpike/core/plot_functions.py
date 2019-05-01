import os
import time
import numpy as np
from core.default_parameters import openGL, gridLines, feature_spike_opacity, feature_spike_size, channel_range, \
    max_spike_plots, max_num_actions
from core.gui_utils import validate_session
from core.Tint_Matlab import find_unit, getspikes, read_cut
from core.feature_functions import CreateFeatures
from core.plot_utils import CustomViewBox, get_channel_color, MultiLine
import pyqtgraph.opengl as gl
import pyqtgraph as pg
# from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
# from core.custom_widgets import GLEllipseROI
from pyqtgraph.Qt import QtCore
from functools import partial
from PyQt5 import QtGui

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


def get_spike_colors(self):
    """
    This function will get the colors of the spike so we can plot them using the same colors that Tint uses.

    :param self: the main window object
    :return: None
    """
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


def get_grid_dimensions(n_cells, method='auto', n=None):
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

        cols = 5
        rows = int(np.ceil(n_cells/cols))
        return rows, cols

    elif method == 'nper':
        if n is None:
            raise ValueError('Invalid n value!')

        else:
            cols = n
            rows = int(np.ceil(n_cells / cols))

        return rows, cols


def drag(self, index, ev=None):
    """
    This function will allow us to modify the line ROI's that we created for each Unity graph. This is so
    the users can select which spikes to cut. We have a dictionary containing each of the ROI's (self.unit_drag_lines),
    where the index (the plot index) is the key, and the value is ROI object.

    Each plot has a different corresponding index as the input to drag, so we can make sure to manipulate the correct
    ROI object using the index input.

    self: the main window object containing the attributes that we will be needing
    index: the index value of the plot (order that the plots were created essentially)
    ev: event.
    """

    # global vb, lr
    if ev.button() == QtCore.Qt.LeftButton:
        # the user is using the left
        if not self.drag_active or index != self.last_drag_index:
            for roi in self.active_ROI:
                try:
                    roi.hide()
                except RuntimeError:
                    # likely the subplot was removed
                    pass

            self.unit_drag_lines[index].show()  # showing the LineSegmentROI
            self.active_ROI = [self.unit_drag_lines[index]]
            self.drag_active = True

        # defining the start of the selected region
        points = [[self.vb[index].mapToView(ev.buttonDownPos()).x(),
                  self.vb[index].mapToView(ev.buttonDownPos()).y()],
                  [self.vb[index].mapToView(ev.pos()).x(),
                   self.vb[index].mapToView(ev.pos()).y()]]

        self.unit_drag_lines[index].setPoints(points)
        self.last_drag_index = index
        ev.accept()
    else:
        pg.ViewBox.mouseDragEvent(self.vb[index], ev)


def getSlope(points):
    points_diffs = np.diff(points, axis=0).flatten()
    slope = points_diffs[1] / points_diffs[0]

    if points_diffs[0] == 0:
        print('hi')

    return slope


def findSpikeSubsample(data, max_n):
    if data.shape[0] > max_n:
        # show the clipped values

        ymax = np.amax(data)
        ymin = np.amin(data)

        # remove the clipped cells first
        data_bool = np.zeros_like(data)
        data_bool[np.where((data >= ymax) | (data <= ymin))] = 1
        data_boolsum = np.sum(data_bool, axis=1)
        clipped_cell_i = np.where(data_boolsum >= 5)[0]
        non_clipped_cell_i = np.setdiff1d(np.arange(data.shape[0] - 1), clipped_cell_i.flatten)

        # find the outlying points
        data_i = np.unique(np.concatenate((np.argmax(data[non_clipped_cell_i, :], axis=0),
                                           np.argmin(data[non_clipped_cell_i, :], axis=0))
                                          ).flatten())

        remaining_index_choices = np.setdiff1d(np.arange(data.shape[0] - 1), data_i)

        # now just take evenly spaced indices of the remaining choices
        data_i = np.concatenate((data_i,
                                 remaining_index_choices[np.linspace(0,
                                                                     len(remaining_index_choices) - 1,
                                                                     num=(max_n - len(data_i))
                                                                     ).astype(int)]))
        data = data[data_i, :]

    else:
        data_i = np.arange(data.shape[0])
    return data, data_i


def getYIntercept(slope, point):
    return point[1] - slope * point[0]


def mouse_click_event(self, index, ev=None):
    """
    This function will override the mouse click event. If the user shift + left mouse clicks on a plot, this will
    launch the PopUpCutWindow (stored as self.PopUpCutWindow), so the user can have a better look at the data.

    If the user middle mouse click it will initiate a cutting of the cell. For this to work the user must have
    had the ROI line intersecting with the data on the plot. It will calculate the slope of the ROI line and
    then determine if this line intersects with any of the data. If the line intersects with the data, it will remove
    the data from this spike, and move it to the self.move_to_channel() attribute from the main Window. It will
    then re-plot both the cell that the spikes were removed from, and the cell tha tthe spikes were moved to.

    :param self: the main window so we can grab any attributes that we need.
    :param index: the index of the plots, essentially the order of which the plots were created
    :param ev: the event that caused this function to run
    :return: None
    """

    # determine if the user used any modifiers (shift, ctrl, etc).
    modifiers = QtGui.QApplication.keyboardModifiers()

    if ev.button() == QtCore.Qt.RightButton:
        # open menu
        pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.LeftButton:
        # we want to make sure the user shift + left clicks to initiate the pop up, check that the modifier is a
        # shift modifier.
        if modifiers == QtCore.Qt.ShiftModifier:
            # then you will launch the popup

            self.PopUpCutWindow.reset_data()  # clears any old data that might have been on the popup
            self.PopUpCutWindow.plot(index, get_cell_from_index(self, index)) # needs index, cell as inputs
        else:
            # hopefully drag event
            pg.ViewBox.mouseClickEvent(self.vb[index], ev)

    elif ev.button() == QtCore.Qt.MiddleButton:
        # then we will accept the changes

        if self.unit_drag_lines[index] in self.active_ROI:
            # then we have an active ROI
            # we will get the x,y positions (rounded to the nearest int) of the selected line

            cell = get_cell_from_index(self, index)

            try:
                invalid_cell_number = int(self.move_to_channel.text())
            except:
                self.choice = None
                self.LogError.signal.emit('InvalidMoveChannel')
                while self.choice is None:
                    time.sleep(0.1)
                return

            if not validateMoveValue(invalid_cell_number):
                self.choice = None
                self.LogError.signal.emit('InvalidMoveChannel')
                while self.choice is None:
                    time.sleep(0.1)
                return

            if invalid_cell_number == cell:
                self.choice = None
                self.LogError.signal.emit('SameChannelInvalid')
                while self.choice is None:
                    time.sleep(0.1)
                return

            points = np.rint(np.asarray(self.unit_drag_lines[index].getState()['points']))

            # find which channel the user started in
            channel = get_channel_from_y(points[0, 1], channel_range=channel_range, n_channels=self.n_channels)

            unit_data = self.unit_data[index][channel]

            crossed_cells = find_spikes_crossed(points, unit_data, samples_per_spike=self.samples_per_spike)

            # append the crossed lines to the invalid cell's plot
            invalid_index = get_index_from_cell(self, invalid_cell_number)

            reconfigure = False
            # remove these spikes from all the channels
            for data_chan in np.arange(self.n_channels):
                if invalid_index is not None:
                    # get the invalid data
                    invalid_cell_data = self.unit_data[index][data_chan][crossed_cells, :]
                    # update the invalid_data channel with this current data
                    self.unit_data[invalid_index][data_chan] = np.vstack((self.unit_data[invalid_index][data_chan],
                                                                          invalid_cell_data))

                    # update the plotted subsample as well
                    _, subsample_i = findSpikeSubsample(self.unit_data[invalid_index][data_chan], max_spike_plots)
                    if invalid_cell_number not in self.cell_subsample_i.keys():
                        self.cell_subsample_i[invalid_cell_number] = {data_chan: subsample_i}
                    else:
                        self.cell_subsample_i[invalid_cell_number][data_chan] = subsample_i
                else:
                    reconfigure = True

                # delete the invalid data from the selected channel
                self.unit_data[index][data_chan] = np.delete(self.unit_data[index][data_chan], crossed_cells, axis=0)

                # recalculate subplot for the channel that the spikes were removed from
                if len(self.unit_data[index][data_chan]) > 0:
                    _, subsample_i = findSpikeSubsample(self.unit_data[index][data_chan], max_spike_plots)
                    if cell not in self.cell_subsample_i.keys():
                        self.cell_subsample_i[cell] = {data_chan: subsample_i}
                    else:
                        self.cell_subsample_i[cell][data_chan] = subsample_i
                else:
                    # there is no data left, don't need to worry about the subsampling anymore
                    if cell in self.cell_subsample_i.keys():
                        self.cell_subsample_i.pop(cell)
                        reconfigure = True

            # check if the cell still exists
            for key, value in self.unit_data[index].items():
                if len(value) == 0:
                    self.unit_data.pop(index)
                    reconfigure = True
                    break

            # update the bool
            cell_indices = self.cell_indices[cell]
            # append invalid cells to the new cell number
            invalid_cells = cell_indices[crossed_cells]
            self.cell_indices[cell] = np.delete(cell_indices, crossed_cells)

            # check if there are still indices for this cell, if empty we will remove
            if len(self.cell_indices[cell]) == 0:
                clear_unit(self, cell)  # delete the cell's plots
                if cell in self.original_cell_count.keys():
                    self.original_cell_count.pop(cell)
                reconfigure = True
                self.cell_indices.pop(cell)

            if invalid_cell_number in self.cell_indices.keys():
                # the cell has existed already within the main window, we can just add to this plot
                self.cell_indices[invalid_cell_number] = np.concatenate((self.cell_indices[invalid_cell_number],
                                                                         invalid_cells))
            else:
                # this cell is not already plotted, have to add the plot and possibly reconfigure
                self.cell_indices[invalid_cell_number] = invalid_cells
                reconfigure = True

            # add the latest action
            if len(self.latest_actions) == 0 or max_num_actions == 1:
                self.latest_actions = {0: {'action': 'cut', 'fromCell': cell, 'toCell': invalid_cell_number,
                                           'movedCutIndices': invalid_cells}}
            else:
                next_action = get_next_action(self)
                self.latest_actions[next_action] = {'action': 'cut', 'fromCell': cell,
                                                    'toChannel': invalid_cell_number, 'toCell': invalid_cells}

            if not reconfigure:
                # update plots for the invalid cell and the cell you removed these spikes from
                # no need to reconfigure
                replot_unit(self, index)
                invalid_index = get_index_from_cell(self, invalid_cell_number)
                replot_unit(self, invalid_index)
            else:
                # we will need to reconfigure the main window possibly, do so

                if cell in self.cell_indices.keys():
                    replot_unit(self, index)

                unique_cells = np.asarray(list(self.cell_indices.keys()))
                reconfigure_units(self, list(unique_cells[unique_cells != 0]))

            if index in self.unit_drag_lines:
                self.unit_drag_lines[index].hide()

            try:
                self.active_ROI.remove(self.unit_drag_lines[index])
            except ValueError:
                pass
            except KeyError:
                pass

            self.drag_active = False
            self.actions_made = True


def get_next_action(self):
    return max(self.latest_actions.keys()) + 1


def get_index_from_cell(self, cell):
    for index, value in self.unit_plots.items():
        if len(value) == 1:
            continue
        else:
            if value[1] == cell:
                return index
    return None


def get_index_from_old_cell(cell, unit_plots):
    for index, value in unit_plots.items():
        if len(value) == 1:
            continue
        else:
            if value[1] == cell:
                return index
    return None


def get_cell_from_index(self, index):
    return self.unit_plots[index][1]


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
    """
    This function will re-plot the cell. This is likely called when you have indicated that you wanted to remove the
    spikes from the cell to another value.

    :param self:
    :param index:
    :param cell:
    :return:
    """

    # skip the 0'th cell, that is the dummy cell in Tint
    if cell == 0:
        return

    # get the cell for the color
    if cell is None:
        cell = get_cell_from_index(self, index)

    setTitle = False
    for channel in np.arange(self.n_channels):
        if index in self.plot_lines.keys():
            self.unit_plots[index][0].removeItem(self.plot_lines[index][channel])
            self.unit_plots[index][0].removeItem(self.avg_plot_lines[index][channel])

        plot_data = self.unit_data[index][channel]
        current_n = plot_data.shape[0]

        self.plot_lines[index][channel] = MultiLine(
            np.tile(np.arange(self.samples_per_spike), (plot_data.shape[0], 1)),
            plot_data, pen_color=get_channel_color(cell))

        self.avg_plot_lines[index][channel] = MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                               np.mean(self.unit_data[index][channel], axis=0).reshape((1, -1)),
                                               pen_color='w', pen_width=2)

        self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
        self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

        if not setTitle:
            if cell in self.original_cell_count.keys():
                setPlotTitle(self.unit_plots[index][0], cell, original_cell_count=self.original_cell_count[cell],
                             current_cell_count=current_n)
            else:
                setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)
            setTitle = True

    if self.PopUpCutWindow.PopUpActive:
        if self.PopUpCutWindow.cell == cell:
            self.PopUpCutWindow.plot(index, cell)


def clear_unit(self, cell):
    """
    This function will clear the plots for a cell that has been removed.

    :param self:
    :param cell:
    :return:
    """

    index = get_index_from_cell(self, cell)

    for chan in np.arange(self.n_channels):
        self.unit_plots[index][0].removeItem(self.plot_lines[index][chan])
        self.unit_plots[index][0].removeItem(self.avg_plot_lines[index][chan])


def reconfigure_units(self, unique_cells):
    """
    This function is called when the cell that you want to plot has not been created yet. Thus we might need to
    reconfigure the subplots (add a row or column).

    :param self:
    :param unique_cells:
    :return:
    """

    unique_cells = sorted(unique_cells)

    # number of cells to plot (this includes the newly added cell)
    n_cells = len(unique_cells)

    # the number of rows and columns that we want to have plotted
    rows, cols = get_grid_dimensions(n_cells, method='nper', n=3)

    # initialize the current row and column count
    row = 0
    col = 0

    # we will create a dictionary to contain the old spike plots
    old_plots = {}
    for index, value in self.plot_lines.items():
        old_plots[index] = value
    self.plot_lines = {}

    # we will create a dictionary to contain the old average plots
    old_avgs = {}
    for index, value in self.avg_plot_lines.items():
        old_avgs[index] = value
    self.avg_plot_lines = {}

    # we will create a dictionary to contain the old unit data
    old_unit_data = {}
    for index, value in self.unit_data.items():
        old_unit_data[index] = value
    self.unit_data = {}

    # we will create a dictionary to contain the old average plots
    old_unit_plots = {}
    old_plotted_units = []
    for index, value in self.unit_plots.items():
        old_unit_plots[index] = value
        old_plotted_units.append(value[1])  # append the cell value
    self.unit_plots = {}

    # get the old positions of the plots (row, col)
    self.old_positions = {}
    for index, value in self.unit_positions.items():
        self.old_positions[index] = value
    self.unit_positions = {}

    # get the old ROI objects
    old_unit_drag_lines = {}
    for index, value in self.unit_drag_lines.items():
        old_unit_drag_lines[index] = value
        value.hide()
    self.unit_drag_lines = {}

    # get the old viewbox objects
    old_vb = {}
    for index, value in self.vb.items():
        old_vb[index] = value
    self.vb = {}

    for index in np.arange(rows*cols):

        if index < len(unique_cells):
            # then we should re-plot these subplot indices

            cell = unique_cells[index]

            # first we need to determine if we need to add a plot or not
            if (row, col) not in self.old_positions.keys():
                '''
                The plot does not exist in this position before, so we will need to add a subplot in this position
                configure the plot appropriately. As well as add another polyline ROI, and define the drag and mouse
                clicks
                '''

                # add the plot
                self.unit_positions[(row, col)] = index
                self.unit_plots[index] = [self.unit_win.addPlot(row=row, col=col,
                                                                viewBox=CustomViewBox(self, self.unit_win)), cell]
                # add the plot
                self.vb[index] = self.unit_plots[index][0].vb

                # configure the plot
                self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
                self.unit_plots[index][0].setYRange(0, -self.n_channels * channel_range, padding=0)
                self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
                self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
                self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
                self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
                self.unit_plots[index][0].enableAutoRange(False, False)

                # add the ROI
                self.unit_drag_lines[index] = pg.PolyLineROI([[0, 0], [30, 30]])
                self.unit_drag_lines[index].hide()
                self.unit_plots[index][0].addItem(self.unit_drag_lines[index])

                # add the drag and mouse click events to the viewbox
                self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
                self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

            else:
                '''
                The row/column does exist. we won't need to create a new subplot, but we will need to move
                everything over and delete the old contents.
                '''
                # we will need to find the old index value so we can make sure we can access that plot
                old_plot_index = get_old_index_from_position(self, (row, col))  # get the index

                # defining the new plot use
                self.unit_plots[index] = [old_unit_plots[old_plot_index][0], cell]
                self.unit_positions[(row, col)] = index

                self.vb[index] = self.unit_plots[index][0].vb
                self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
                self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

                self.unit_drag_lines[index] = old_unit_drag_lines[old_plot_index]

            if cell in old_plotted_units:
                # figure out if we have plotted this cell before, if so, we can move that data instead of re-plotting

                # if index < len(old_indices):
                # if the index is less than the length of the old indices, then

                # old_index = old_indices[index]
                old_index = get_index_from_old_cell(cell, old_unit_plots)

                current_n = len(self.cell_indices[cell])
                if cell in self.original_cell_count.keys():
                    setPlotTitle(self.unit_plots[index][0], cell,
                                 original_cell_count=self.original_cell_count[cell],
                                 current_cell_count=current_n)
                else:
                    setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)

                # we are defining the new spike plots and avg plots given the old index
                self.plot_lines[index] = old_plots[old_index]
                self.avg_plot_lines[index] = old_avgs[old_index]

                # moving the line plots
                for channel in self.plot_lines[index].keys():
                    old_unit_plots[old_index][0].removeItem(old_plots[old_index][channel])
                    self.unit_plots[index][0].addItem(self.plot_lines[index][channel])

                # moving the avg plots
                for channel in self.avg_plot_lines[index].keys():
                    old_unit_plots[old_index][0].removeItem(old_avgs[old_index][channel])
                    self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])

                # moving the unit data
                self.unit_data[index] = old_unit_data[old_index]

            else:
                # This cell has not been plotted before, so we will need to create the MultiLines

                setTitle = False
                for channel in np.arange(self.n_channels):

                    self.unit_positions[(row, col)] = index

                    # shifting the data so that the next channel resides below the previous
                    # also making the 1st channel start at a y value of 0

                    cell_bool = self.cell_indices[cell]
                    cell_data = self.tetrode_data[:, cell_bool, :]

                    plot_data = cell_data[channel]
                    channel_max = np.amax(plot_data)
                    plot_data = plot_data - channel * channel_range - channel_max

                    current_n = plot_data.shape[0]
                    if not setTitle:
                        if cell in self.original_cell_count.keys():
                            setPlotTitle(self.unit_plots[index][0], cell,
                                         original_cell_count=self.original_cell_count[cell],
                                         current_cell_count=current_n)
                        else:
                            setPlotTitle(self.unit_plots[index][0], cell, current_cell_count=current_n)
                        setTitle = True

                    cell_data_sub_channel, subsample_i = findSpikeSubsample(plot_data, max_spike_plots)

                    if cell not in self.cell_subsample_i.keys():
                        self.cell_subsample_i[cell] = {channel: subsample_i}
                    else:
                        self.cell_subsample_i[cell][channel] = subsample_i

                    plot_data_avg = np.mean(plot_data, axis=0).reshape((1, -1))

                    if index not in self.unit_data.keys():
                        self.unit_data[index] = {channel: plot_data}
                    else:
                        self.unit_data[index][channel] = plot_data

                    if index not in self.plot_lines.keys():

                        self.plot_lines[index] = {
                            channel: MultiLine(
                                np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                cell_data_sub_channel, pen_color=get_channel_color(cell))}
                    else:
                        self.plot_lines[index][channel] = MultiLine(
                            np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                            cell_data_sub_channel, pen_color=get_channel_color(cell))

                    if index not in self.avg_plot_lines.keys():
                        self.avg_plot_lines[index] = {
                            channel: MultiLine(np.arange(self.samples_per_spike).reshape((1, -1)),
                                               plot_data_avg, pen_color='w', pen_width=2)}
                    else:
                        self.avg_plot_lines[index][channel] = MultiLine(
                            np.arange(self.samples_per_spike).reshape((1, -1)),
                            plot_data_avg, pen_color='w', pen_width=2)

                    self.unit_plots[index][0].addItem(self.plot_lines[index][channel])
                    self.unit_plots[index][0].addItem(self.avg_plot_lines[index][channel])
        else:
            # i made this to remove the items from the plots that we don't need anymore, however I end up just
            # removing the subplot itself so this is no longer necessary
            '''
            # then we should check if the plot has any data on it, and remove it since we no longer need it
            if (row, col) in self.old_positions.keys():
                # then we had previously plotted stuff here
                old_plot_index = get_old_index_from_position(self, (row, col))  # get the index

                for channel in old_plots[old_plot_index].keys():
                    old_unit_plots[old_plot_index][0].removeItem(old_plots[old_plot_index][channel])

                # removing the avg plots
                for channel in old_avgs[old_plot_index].keys():
                    old_unit_plots[old_plot_index][0].removeItem(old_avgs[old_plot_index][channel])

            else:
                # there is nothing plotted, so don't worry about it
                pass
            '''
            pass

        col += 1
        if col >= cols:
            col = 0
            row += 1

    # removing any sub_plots that are no longer necessary
    for position in self.old_positions.keys():
        if position not in self.unit_positions.keys():
            old_plot_index = get_old_index_from_position(self, position)  # get the i
            plot_item = old_unit_plots[old_plot_index][0]
            self.unit_win.removeItem(plot_item)

    self.old_positons = {}
    self.unit_rows = rows
    self.unit_cols = cols


def setPlotTitle(plot, cell, original_cell_count=None, current_cell_count=None):
    """
    This function will set the title for each of the plots so we know which cell is being plotted.

    :param plot:
    :param cell:
    :param original_cell_count:
    :param current_cell_count:
    :return:
    """
    title_text = "Cell %d, " % cell
    if current_cell_count is not None:
        title_text += "%d spikes" % current_cell_count
    if original_cell_count is not None:
        percent_change = np.abs(100 * (original_cell_count - current_cell_count) / original_cell_count)
        if current_cell_count > original_cell_count:
            title_text += " <span style='color:green'>(add %.2f%%)</span>" % percent_change
        elif current_cell_count < original_cell_count:
            title_text += " <span style='color:red'>(rem %.2f%%)</span>" % percent_change
    plot.setTitle(title_text)


def get_index_from_position(self, position):
    if position in self.unit_positions.keys():
        return self.unit_positions[position]
    return None


def get_old_index_from_position(self, position):
    if position in self.old_positions.keys():
        return self.old_positions[position]
    return None


def get_position_from_index(self, index):
    for position, index_ in self.unit_positions.items():
        if index == index_:
            return position
    return None


def add_graph_limits():
    pass


def plot_units(self):

    unique_cells = np.unique(self.cut_data)
    # the 0'th cell is the dummy cell for Tint so we will remove that

    unique_cells = unique_cells[unique_cells != 0]

    self.cell_indices[0] = np.where(self.cut_data == 0)[0]

    n_cells = len(unique_cells)

    rows, cols = get_grid_dimensions(n_cells, method='nper', n=3)

    row = 0
    col = 0
    for index in np.arange(len(unique_cells)):
        self.unit_positions[(row, col)] = index

        cell = unique_cells[index]

        self.unit_plots[index] = [self.unit_win.addPlot(row=row, col=col,
                                                        viewBox=CustomViewBox(self, self.unit_win)), cell]

        self.vb[index] = self.unit_plots[index][0].vb

        self.unit_plots[index][0].setXRange(0, self.samples_per_spike, padding=0)  # set the x-range
        self.unit_plots[index][0].setYRange(0, -self.n_channels * channel_range, padding=0)
        self.unit_plots[index][0].hideAxis('left')  # remove the y-axis
        self.unit_plots[index][0].hideAxis('bottom')  # remove the x axis
        self.unit_plots[index][0].hideButtons()  # hide the auto-resize button
        self.unit_plots[index][0].setMouseEnabled(x=False, y=False)  # disables the mouse interactions
        self.unit_plots[index][0].enableAutoRange(False, False)

        self.unit_drag_lines[index] = pg.PolyLineROI([[0, 0], [30, 30]])
        self.unit_drag_lines[index].hide()
        self.unit_plots[index][0].addItem(self.unit_drag_lines[index])

        self.vb[index].mouseDragEvent = partial(drag, self, index)  # overriding the drag event
        self.vb[index].mouseClickEvent = partial(mouse_click_event, self, index)

        cell_bool = np.where(self.cut_data == cell)[0]
        cell_data = self.tetrode_data[:, cell_bool, :]

        self.original_cell_count[cell] = cell_data.shape[1]

        if self.samples_per_spike is None:
            self.samples_per_spike = cell_data.shape[2]

        if self.n_channels is None:
            self.n_channels = cell_data.shape[0]

        self.cell_indices[cell] = cell_bool

        # cell_data_sub = findSpikeSubsample(cell_data, max_spike_plots)

        title_set = False
        for channel in np.arange(self.n_channels):
            # we will keep the data and the indices

            # shifting the data so that the next channel resides below the previous
            # also making the 1st channel start at a y value of 0
            plot_data = cell_data[channel]
            channel_max = np.amax(plot_data)
            plot_data = plot_data - channel*channel_range - channel_max
            if not title_set:
                setPlotTitle(self.unit_plots[index][0], cell, original_cell_count=self.original_cell_count[cell],
                             current_cell_count=plot_data.shape[0])
                title_set = True

            cell_data_sub_channel, subsample_i = findSpikeSubsample(plot_data, max_spike_plots)

            if cell not in self.cell_subsample_i.keys():
                self.cell_subsample_i[cell] = {channel: subsample_i}
            else:
                self.cell_subsample_i[cell][channel] = subsample_i

            plot_data_avg = np.mean(plot_data, axis=0).reshape((1, -1))

            if index not in self.unit_data.keys():
                self.unit_data[index] = {channel: plot_data}
            else:
                self.unit_data[index][channel] = plot_data

            if index not in self.plot_lines.keys():

                self.plot_lines[index] = {channel: MultiLine(np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                                             cell_data_sub_channel, pen_color=get_channel_color(cell))}
            else:
                self.plot_lines[index][channel] = MultiLine(np.tile(np.arange(self.samples_per_spike), (cell_data_sub_channel.shape[0], 1)),
                                                            cell_data_sub_channel, pen_color=get_channel_color(cell))

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

    self.reset_parameters()

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
                cut_data = read_cut(self.cut_filename.text())
                # cut_data = find_unit([tetrode_filename])
                # self.cut_data = cut_data[0]
                self.cut_data = cut_data
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


def moveToChannel(self, origin):

    if origin == 'main':
        self.PopUpCutWindow.move_to_channel.setText(self.move_to_channel.text())
    elif origin == 'popup':
        self.mainWindow.move_to_channel.setText(self.move_to_channel.text())


def validateMoveValue(channel_value):
    """
    Determine if the channel value is valid or not
    :param channel_value:
    :return:
    """
    if channel_value < 0:
        return False
    elif channel_value > 30:
        return False
    return True


def undo_function(self):
    latest_action_key = max(self.latest_actions.keys())

    latest_action = self.latest_actions[latest_action_key]

    if latest_action['action'] == 'cut':
        # here we will have to undo a movement of spikes from one cell to another
        fromCell = latest_action['fromCell']  # the spikes where moved from here originally
        toCell = latest_action['toCell']  # the spikes were moved here in this action
        cut_indices = latest_action['movedCutIndices']  # the cut file indices that were moved in the transfer

        # identify the spikes that we moved to the new cell
        movedBool = np.where(np.isin(self.cell_indices[toCell], cut_indices))[0]

        toCellIndex = get_index_from_cell(self, toCell)
        fromCellIndex = get_index_from_cell(self, fromCell)

        reconfigure = False
        for data_chan in np.arange(self.n_channels):
            # add the spike data back to it's original channel
            if fromCellIndex is not None:
                # get the moved data
                moved_cell_data = self.unit_data[toCellIndex][data_chan][movedBool, :]
                # put this data back into the fromCell
                self.unit_data[fromCellIndex][data_chan] = np.vstack((self.unit_data[fromCellIndex][data_chan],
                                                                      moved_cell_data))

                # update the plotted subsample as well
                _, subsample_i = findSpikeSubsample(self.unit_data[fromCellIndex][data_chan], max_spike_plots)
                if fromCell not in self.cell_subsample_i.keys():
                    self.cell_subsample_i[fromCell] = {data_chan: subsample_i}
                else:
                    self.cell_subsample_i[fromCell][data_chan] = subsample_i
            else:
                # we don't need to worry about this because the reconfigure function will take care of it
                reconfigure = True

            # remove the data from the channel it was original moved to
            self.unit_data[toCellIndex][data_chan] = np.delete(self.unit_data[toCellIndex][data_chan],
                                                               movedBool, axis=0)

            if len(self.unit_data[toCellIndex][data_chan]) > 0:
                # update the toCell subsample_i as well
                _, subsample_i = findSpikeSubsample(self.unit_data[toCellIndex][data_chan], max_spike_plots)
                if toCell not in self.cell_subsample_i.keys():
                    self.cell_subsample_i[toCell] = {data_chan: subsample_i}
                else:
                    self.cell_subsample_i[toCell][data_chan] = subsample_i
            else:
                # there is no data left, don't need to worry about the subsampling anymore
                if toCell in self.cell_subsample_i.keys():
                    self.cell_subsample_i.pop(toCell)
                    reconfigure = True

        # this cell no longer exists
        for key, value in self.unit_data[toCellIndex].items():
            if len(value) == 0:
                self.unit_data.pop(toCellIndex)
                reconfigure = True
                break

        # make sure that the spike indices are also moved back to their original cell
        cell_indices = self.cell_indices[toCell]
        undo_cells = cell_indices[movedBool]
        self.cell_indices[toCell] = np.delete(cell_indices, movedBool)

        # remove the cell indices key for this cell if it has no data anymore
        if len(self.cell_indices[toCell]) == 0:
            reconfigure = True
            self.cell_indices.pop(toCell)

        # add the spikes back to where they used to be
        # determine if we need to reconfigure the main window

        if fromCell in self.cell_indices.keys():
            # the cell has existed already within the main window, we can just add to this plot
            self.cell_indices[fromCell] = np.concatenate((self.cell_indices[fromCell], undo_cells))
        else:
            # this cell is not already plotted, have to add the plot and possibly reconfigure
            self.cell_indices[fromCell] = undo_cells
            reconfigure = True

        # plot the data
        if not reconfigure:
            # update plots for the invalid cell and the cell you removed these spikes from
            # no need to reconfigure
            replot_unit(self, toCellIndex)
            replot_unit(self, fromCellIndex)
        else:
            # we will need to reconfigure the main window possibly, do so
            replot_unit(self, fromCellIndex)
            unique_cells = np.asarray(list(self.cell_indices.keys()))
            reconfigure_units(self, list(unique_cells[unique_cells != 0]))

        # remove this action from the dictionary
        self.latest_actions.pop(latest_action_key)

    else:
        print('The following action has not been coded yet: %s' % latest_action['action'])