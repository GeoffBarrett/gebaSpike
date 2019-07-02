import numpy as np
import time


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

        if data_i.shape[0] < max_n:

            remaining_index_choices = np.setdiff1d(np.arange(data.shape[0] - 1), data_i)

            # now just take evenly spaced indices of the remaining choices
            data_i = np.concatenate((data_i,
                                     remaining_index_choices[np.linspace(0,
                                                                         len(remaining_index_choices) - 1,
                                                                         num=(max_n - len(data_i))
                                                                         ).astype(int)]))
        elif data_i.shape[0] > max_n:
            data_i = data_i[:max_n]

        data = data[data_i, :]

    else:
        data_i = np.arange(data.shape[0])
    return data, data_i


def getYIntercept(slope, point):
    return point[1] - slope * point[0]


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


def validateMaxSpikes(max_spikes):
    """
    :param channel_value:
    :return:
    """
    try:
        max_spike = int(max_spikes)
        if max_spike < 0:
            return False
    except:
        return False

    return True


def get_max_spikes(self):
    max_spike_plots = self.max_spike_plots_text.text()
    max_spike_valid = validateMaxSpikes(max_spike_plots)
    if max_spike_valid:
        max_spike_plots = int(max_spike_plots)
    else:
        self.choice = None
        self.LogError.signal.emit('invalidMaxSpikes')
        while self.choice is None:
            time.sleep(0.1)
        return None

    return max_spike_plots


def get_next_action(self):
    return max(self.latest_actions.keys()) + 1


def clear_unit(self, cell):
    """
    This function will clear the plots for a cell that has been removed.

    :param self:
    :param cell:
    :return:
    """

    if cell == 0:
        return

    index = get_index_from_cell(self, cell)

    for chan in np.arange(self.n_channels):
        self.unit_plots[index][0].removeItem(self.plot_lines[index][chan])
        self.unit_plots[index][0].removeItem(self.avg_plot_lines[index][chan])


def moveToChannel(self, origin):

    if origin == 'main':
        for key in self.PopUpCutWindow.keys():
            self.PopUpCutWindow[key].move_to_channel.setText(self.move_to_channel.text())
    elif origin == 'popup':

        cell = self.cell
        # for key in self.
        self.mainWindow.move_to_channel.setText(self.move_to_channel.text())


def maxSpikesChange(self, origin):

    if origin == 'main':
        for key in self.PopUpCutWindow.keys():
            self.PopUpCutWindow[key].max_spike_plots_text.setText(self.max_spike_plots_text.text())
    elif origin == 'popup':
        cell = self.cell
        self.mainWindow.max_spike_plots_text.setText(self.max_spike_plots_text.text())
    self.max_spike_plots = None  # we will set this value to none as to re-validate the value
