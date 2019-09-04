"""
I have recently combined the two methods for cutting the cell (the main window vs the pop up window). This should make
it easier to add and optimize features since the code will be in the same place. However I left the legacy code here
 just in case (although theoretically it will be on GitHub).

"""

def mouse_click_eventPopup(self, vb, ev=None):

    if ev.button() == QtCore.Qt.RightButton:
        # open menu
        pg.ViewBox.mouseClickEvent(vb, ev)

    elif ev.button() == QtCore.Qt.LeftButton:

        # hopefully drag event
        pg.ViewBox.mouseClickEvent(vb, ev)

    elif ev.button() == QtCore.Qt.MiddleButton:
        # then we will accept the changes

        if self.index is None:
            return

        if self.unit_drag_lines in self.active_ROI or self.channel_drag_lines in self.active_ROI:
            # then we have an active ROI
            # we will get the x,y positions (rounded to the nearest int) of the selected line

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

            if invalid_cell_number == self.cell:
                self.choice = None
                self.LogError.signal.emit('SameChannelInvalid')
                while self.choice is None:
                    time.sleep(0.1)
                return

            if self.unit_drag_lines in self.active_ROI:
                points = np.rint(np.asarray(self.unit_drag_lines.getState()['points']))
                channel = get_channel_from_y(points[0, 1], channel_range=channel_range, n_channels=self.n_channels)
            elif self.channel_drag_lines in self.active_ROI:
                points = np.rint(np.asarray(self.channel_drag_lines.getState()['points']))
                channel = int(self.channel_number.currentText()) - 1

            max_spikes_changed = False
            if self.mainWindow.max_spike_plots is None:
                max_spike_plots = get_max_spikes(self.mainWindow)
                if max_spike_plots is None:
                    return
                else:
                    self.mainWindow.max_spike_plots = max_spike_plots

                    # update the max spikes by re-calculating the subsample index values
                    for plotted_cell in self.mainWindow.cell_subsample_i.keys():
                        plotted_channel_index = get_index_from_cell(self.mainWindow, plotted_cell)

                        if any(plotted_cell == cell_value for cell_value in [0]):
                            # skip the invalid channel (done later) and the dummy channel (not plotted)
                            continue

                        for data_chan in np.arange(self.mainWindow.n_channels):
                            _, subsample_i = findSpikeSubsample(self.mainWindow.unit_data[plotted_channel_index][data_chan],
                                                                self.mainWindow.max_spike_plots)

                            if plotted_cell not in self.mainWindow.cell_subsample_i.keys():
                                self.mainWindow.cell_subsample_i[plotted_cell] = {data_chan: subsample_i}
                            else:
                                self.mainWindow.cell_subsample_i[plotted_cell][data_chan] = subsample_i
                    max_spikes_changed = True

            unit_data = self.mainWindow.unit_data[self.index][channel]
            crossed_cells = find_spikes_crossed(points, unit_data, samples_per_spike=self.samples_per_spike)

            invalid_index = get_index_from_cell(self.mainWindow, invalid_cell_number)
            # remove these spikes from all the channels
            reconfigure = False

            if len(crossed_cells) != 0:
                for data_chan in np.arange(self.mainWindow.n_channels):
                    # append the crossed lines to the invalid cell's plot
                    if invalid_index is not None:
                        # get the invalid data
                        invalid_cell_data = self.mainWindow.unit_data[self.index][data_chan][crossed_cells, :]
                        # update the invalid_data channel with this current data
                        self.mainWindow.unit_data[invalid_index][data_chan] = np.vstack((
                            self.mainWindow.unit_data[invalid_index][data_chan], invalid_cell_data))

                        # update the plotted subsample as well
                        _, subsample_i = findSpikeSubsample(self.mainWindow.unit_data[invalid_index][data_chan], self.mainWindow.max_spike_plots)
                        if invalid_cell_number not in self.mainWindow.cell_subsample_i.keys():
                            self.mainWindow.cell_subsample_i[invalid_cell_number] = {data_chan: subsample_i}
                        else:
                            self.mainWindow.cell_subsample_i[invalid_cell_number][data_chan] = subsample_i
                    else:
                        reconfigure = True

                    self.mainWindow.unit_data[self.index][data_chan] = np.delete(
                        self.mainWindow.unit_data[self.index][data_chan], crossed_cells, axis=0)

                    # recalculate subplot for the channel that the spikes were removed from
                    if len(self.mainWindow.unit_data[self.index][data_chan]) > 0:
                        _, subsample_i = findSpikeSubsample(self.mainWindow.unit_data[self.index][data_chan], self.mainWindow.max_spike_plots)
                        if self.cell not in self.mainWindow.cell_subsample_i.keys():
                            self.mainWindow.cell_subsample_i[self.cell] = {data_chan: subsample_i}
                        else:
                            self.mainWindow.cell_subsample_i[self.cell][data_chan] = subsample_i
                    else:
                        # there is no data left, don't need to worry about the subsampling anymore
                        if self.cell in self.mainWindow.cell_subsample_i.keys():
                            self.mainWindow.cell_subsample_i.pop(self.cell)
                            reconfigure = True

                # check if the cell still exists
                for key, value in self.mainWindow.unit_data[self.index].items():
                    if len(value) == 0:
                        if self.index != -1:
                            self.mainWindow.unit_data.pop(self.index)
                            reconfigure = True
                            break

                # update the bool
                cell_indices = self.mainWindow.cell_indices[self.cell]

                # append invalid cells to the new cell number
                invalid_cells = cell_indices[crossed_cells]
                self.mainWindow.cell_indices[self.cell] = np.delete(cell_indices, crossed_cells)

                # check if there are still indices for this cell, if empty we will remove
                if len(self.mainWindow.cell_indices[self.cell]) == 0:
                    reconfigure = True
                    clear_unit(self.mainWindow, self.cell)  # delete the cell's plots
                    if self.cell in self.mainWindow.original_cell_count.keys():
                        self.mainWindow.original_cell_count.pop(self.cell)
                    if self.cell != 0:
                        self.mainWindow.cell_indices.pop(self.cell)

                if invalid_cell_number in self.mainWindow.cell_indices.keys():
                    # the cell has existed already within the main window, we can just add to this plot
                    self.mainWindow.cell_indices[invalid_cell_number] = np.concatenate((
                        self.mainWindow.cell_indices[invalid_cell_number], invalid_cells))
                else:
                    # this cell is not already plotted, have to add the plot and possibly reconfigure
                    self.mainWindow.cell_indices[invalid_cell_number] = invalid_cells

                    if invalid_cell_number != 0:
                        reconfigure = True

                # add the latest action
                if len(self.mainWindow.latest_actions) == 0 or max_num_actions == 1:
                    self.mainWindow.latest_actions = {0: {'action': 'cut', 'fromCell': self.cell,
                                                          'toCell': invalid_cell_number, 'movedCutIndices': invalid_cells}}
                else:
                    next_action = get_next_action(self.mainWindow)
                    self.mainWindow.latest_actions[next_action] = {'action': 'cut', 'fromCell': self.cell,
                                                                   'toCell': invalid_cell_number,
                                                                   'movedCutIndices': invalid_cells}

            # re-plot on popup
            self.plot(self.index, self.cell)
            for roi in self.active_ROI:
                roi.hide()

            # re-plot the main Window
            if not reconfigure:
                # update plots for the invalid cell and the cell you removed these spikes from
                # no need to reconfigure
                replot_unit(self.mainWindow, self.index)
                invalid_index = get_index_from_cell(self.mainWindow, invalid_cell_number)
                if invalid_index != -1:
                    replot_unit(self.mainWindow, invalid_index)
            else:
                # we will need to reconfigure the main window possibly, do so

                if self.cell in self.mainWindow.cell_indices.keys():
                    replot_unit(self.mainWindow, self.index)
                unique_cells = np.asarray(list(self.mainWindow.cell_indices.keys()))
                reconfigure_units(self.mainWindow, list(unique_cells[unique_cells != 0]))

            if max_spikes_changed:
                for plotted_cell in self.mainWindow.cell_indices.keys():
                    if plotted_cell == invalid_cell_number or plotted_cell == self.cell:
                        continue

                    plotted_cell_index = get_index_from_cell(self.mainWindow, plotted_cell)
                    replot_unit(self.mainWindow, plotted_cell_index)

            self.drag_active = False
            self.mainWindow.actions_made = True


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

            cell = get_cell_from_index(self, index)

            if cell not in self.PopUpCutWindow.keys():
                # this cell does not have a popup window, create one
                self.addPopup(cell)
            else:
                # reset the data for this cell
                self.PopUpCutWindow[cell].reset_data()  # clears any old data that might have been on the popup
            self.PopUpCutWindow[cell].plot(index, get_cell_from_index(self, index))  # needs index, cell as inputs
            self.PopUpCutWindow[cell].raise_()

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

            max_spikes_changed = False
            # self.max_spike_plots will be none if the value has been changed recently
            if self.max_spike_plots is None:
                max_spike_plots = get_max_spikes(self)
                if max_spike_plots is None:
                    return
                else:
                    self.max_spike_plots = max_spike_plots

                    # update the max spikes by re-calculating the subsample index values
                    for plotted_cell in self.cell_subsample_i.keys():
                        plotted_channel_index = get_index_from_cell(self, plotted_cell)

                        if any(plotted_cell == cell_value for cell_value in [0]):
                            # skip the invalid channel (done later) and the dummy channel (not plotted)
                            continue

                        for data_chan in np.arange(self.n_channels):
                            _, subsample_i = findSpikeSubsample(self.unit_data[plotted_channel_index][data_chan],
                                                                self.max_spike_plots)

                            if plotted_cell not in self.cell_subsample_i.keys():
                                self.cell_subsample_i[plotted_cell] = {data_chan: subsample_i}
                            else:
                                self.cell_subsample_i[plotted_cell][data_chan] = subsample_i
                    max_spikes_changed = True

            reconfigure = False

            if len(crossed_cells) != 0:
                # remove these spikes from all the channels
                for data_chan in np.arange(self.n_channels):
                    if invalid_index is not None:
                        # get the invalid data
                        invalid_cell_data = self.unit_data[index][data_chan][crossed_cells, :]
                        # update the invalid_data channel with this current data
                        self.unit_data[invalid_index][data_chan] = np.vstack((self.unit_data[invalid_index][data_chan],
                                                                              invalid_cell_data))

                        # update the plotted subsample as well
                        _, subsample_i = findSpikeSubsample(self.unit_data[invalid_index][data_chan], self.max_spike_plots)
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
                        _, subsample_i = findSpikeSubsample(self.unit_data[index][data_chan], self.max_spike_plots)
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
                        if index != -1:
                            # avoid popping the dummy cell (index -1)
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

                    if cell != 0:
                        self.cell_indices.pop(cell)

                if invalid_cell_number in self.cell_indices.keys():
                    # the cell has existed already within the main window, we can just add to this plot
                    self.cell_indices[invalid_cell_number] = np.concatenate((self.cell_indices[invalid_cell_number],
                                                                             invalid_cells))
                else:
                    # this cell is not already plotted, have to add the plot and possibly reconfigure
                    self.cell_indices[invalid_cell_number] = invalid_cells

                    if invalid_cell_number != 0:
                        reconfigure = True

                # add the latest action
                if len(self.latest_actions) == 0 or max_num_actions == 1:
                    self.latest_actions = {0: {'action': 'cut', 'fromCell': cell, 'toCell': invalid_cell_number,
                                               'movedCutIndices': invalid_cells}}
                else:
                    next_action = get_next_action(self)
                    self.latest_actions[next_action] = {'action': 'cut', 'fromCell': cell, 'toCell': invalid_cell_number,
                                                        'movedCutIndices': invalid_cells}

            if not reconfigure:
                # update plots for the invalid cell and the cell you removed these spikes from
                # no need to reconfigure
                replot_unit(self, index)
                invalid_index = get_index_from_cell(self, invalid_cell_number)
                if invalid_index != -1:
                    replot_unit(self, invalid_index)
            else:
                # we will need to reconfigure the main window possibly, do so

                if cell in self.cell_indices.keys():
                    replot_unit(self, index)

                unique_cells = np.asarray(list(self.cell_indices.keys()))
                reconfigure_units(self, list(unique_cells[unique_cells != 0]))

            if max_spikes_changed:
                for plotted_cell in self.cell_indices.keys():
                    if plotted_cell == invalid_cell_number or plotted_cell == cell:
                        continue

                    plotted_cell_index = get_index_from_cell(self, plotted_cell)
                    replot_unit(self, plotted_cell_index)

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
