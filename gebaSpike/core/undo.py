import numpy as np
from .waveform_cut_functions import get_index_from_cell, get_max_spikes, findSpikeSubsample
from .plot_functions import replot_unit, reconfigure_units


def undo_function(self):

    if len(self.latest_actions) == 0:
        return

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

        if self.max_spike_plots is None:
            max_spike_plots = get_max_spikes(self)
            if max_spike_plots is None:
                return
            else:
                self.max_spike_plots = max_spike_plots

        reconfigure = False
        for data_chan in np.arange(self.n_channels):
            # add the spike data back to it's original channel

            # get the moved data
            moved_cell_data = self.unit_data[toCellIndex][data_chan][movedBool, :]

            if fromCellIndex is not None:
                # put this data back into the fromCell
                self.unit_data[fromCellIndex][data_chan] = np.vstack((self.unit_data[fromCellIndex][data_chan],
                                                                      moved_cell_data))

                # update the plotted subsample as well
                _, subsample_i = findSpikeSubsample(self.unit_data[fromCellIndex][data_chan], self.max_spike_plots)
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
                _, subsample_i = findSpikeSubsample(self.unit_data[toCellIndex][data_chan], self.max_spike_plots)
                if toCell not in self.cell_subsample_i.keys():
                    self.cell_subsample_i[toCell] = {data_chan: subsample_i}
                else:
                    self.cell_subsample_i[toCell][data_chan] = subsample_i
            else:
                # there is no data left, don't need to worry about the sub-sampling anymore
                if toCell in self.cell_subsample_i.keys():
                    self.cell_subsample_i.pop(toCell)
                    reconfigure = True

        # this cell no longer exists
        for key, value in self.unit_data[toCellIndex].items():
            if len(value) == 0:
                if toCellIndex != -1:
                    # -1 is the index of the dummy cell (0), lets make sure not to remove this
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
            if toCell != 0:
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
            if toCellIndex != -1:
                replot_unit(self, toCellIndex)

            replot_unit(self, fromCellIndex)
        else:
            # we will need to reconfigure the main window possibly, do so

            if fromCellIndex in self.unit_plots.keys():
                replot_unit(self, fromCellIndex)
            unique_cells = np.asarray(list(self.cell_indices.keys()))
            reconfigure_units(self, list(unique_cells[unique_cells != 0]))

        # remove this action from the dictionary
        self.latest_actions.pop(latest_action_key)

    else:
        print('The following action has not been coded yet: %s' % latest_action['action'])