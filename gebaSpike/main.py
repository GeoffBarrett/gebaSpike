import sys
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from core.gui_utils import center, validate_session, Communicate, validate_cut
from core.default_parameters import project_name, default_filename, defaultXAxis, defaultYAxis, defaultZAxis, openGL, \
default_move_channel
from core.Tint_Matlab import find_tetrodes
from core.plot_functions import manage_features, feature_name_map, moveToChannel, undo_function
from core.PopUpCutting import PopUpCutWindow
from core.cut_functions import write_cut
from core.plot_functions import get_index_from_cell
import pyqtgraph.opengl as gl
import os
import json
import time
import numpy as np


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("%s - Main Window" % project_name)  # sets the main window title

        # initializing attributes
        self.cut_filename = None
        self.choose_cut_filename_btn = None
        self.feature_win = None
        self.quit_btn = None
        self.filename = None
        self.choose_filename_btn = None
        self.x_axis_cb = None
        self.y_axis_cb = None
        self.z_axis_cb = None
        self.tetrode_cb = None
        self.choice = None
        self.plot_btn = None
        self.feature_plot = None

        self.latest_actions = {}

        self.actions_made = False

        self.drag_active = False

        self.last_drag_index = None

        self.feature_data = None
        self.tetrode_data = None
        self.cut_data = None
        self.cut_data_original = None
        self.spike_times = None
        self.scatterItem = None
        self.glViewWidget = None
        self.feature_plot_added = False
        self.samples_per_spike = None

        self.unit_positions = {}
        self.cell_subsample_i = {}
        self.unit_drag_lines = {}
        self.active_ROI = []
        self.unit_data = {}

        self.xline = None
        self.yline = None
        self.zline = None

        self.n_channels = None

        self.invalid_channel = None

        self.cell_indices = {}

        self.spike_colors = None

        self.original_cell_count = {}

        self.unit_plots = {}
        self.vb = {}
        self.plot_lines = {}
        self.avg_plot_lines = {}
        self.unit_rows = 0
        self.unit_cols = 0

        self.LogError = Communicate()
        self.LogError.signal.connect(self.raiseError)

        # various setting files
        project_dir = os.path.dirname(os.path.abspath("__file__"))
        if os.path.basename(project_dir) != project_name:
            project_dir = os.path.dirname(sys.argv[0])

        # defining the directory filepath
        self.PROJECT_DIR = project_dir  # project directory
        self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')  # settings directory

        self.settings_filename = os.path.join(self.SETTINGS_DIR, 'settings.json')
        self.settings = self.get_settings()

        self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')  # settings directory

        # create Settings Directory
        if not os.path.exists(self.SETTINGS_DIR):
            os.mkdir(self.SETTINGS_DIR)

        # create settings file
        if not os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'w') as f:
                json.dump({}, f)

        # setting PyQtGraph settings
        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        self.PopUpCutWindow = PopUpCutWindow(self)

        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('GTK+'))

        self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'GEBA_Logo.png')))  # declaring the icon image
        self.initialize()  # initializes the main window

    def initialize(self):
        """
        This method will initialize the Main Window, create all the widgets and what not
        :return:
        """

        # ---------------- session filename options ------------------------------------------

        filename_grid_layout = QtWidgets.QGridLayout()

        filename_label = QtWidgets.QLabel("Filename:")
        self.filename = QtWidgets.QLineEdit()
        self.filename.setText(default_filename)
        self.filename.textChanged.connect(self.filename_changed)
        self.filename.setToolTip('The name of the session that you will analyze in gebaSpike!')
        self.choose_filename_btn = QtWidgets.QPushButton("Choose Filename")
        self.choose_filename_btn.setToolTip('This button will allow you to choose the filename to analyze!')
        self.choose_filename_btn.clicked.connect(self.choose_filename)

        filename_grid_layout.addWidget(self.choose_filename_btn, *(0, 0))
        filename_grid_layout.addWidget(filename_label, *(0, 1), QtCore.Qt.AlignRight)

        # ------------------- cut filename options ---------------------------------

        cut_filename_label = QtWidgets.QLabel("Cut Filename:")
        self.cut_filename = QtWidgets.QLineEdit()
        self.cut_filename.setText(default_filename)
        # self.cut_filename.textChanged.connect(self.cut_filename_changed)
        self.cut_filename.setToolTip('The name of the cut file containing the sorted values!')
        self.choose_cut_filename_btn = QtWidgets.QPushButton("Choose Cut Filename")
        self.choose_cut_filename_btn.setToolTip('This button will allow you to choose a cut file to use!')
        self.choose_cut_filename_btn.clicked.connect(self.choose_cut_filename)

        line_edit_layout = QtWidgets.QVBoxLayout()
        line_edit_layout.addWidget(self.filename)
        line_edit_layout.addWidget(self.cut_filename)

        filename_grid_layout.addWidget(self.choose_cut_filename_btn, *(1, 0))
        filename_grid_layout.addWidget(cut_filename_label, *(1, 1), QtCore.Qt.AlignRight)

        filename_grid_layout.addLayout(line_edit_layout, *(0, 2), 2, 1)

        # ------- move invalid -----------

        move_to_channel_label = QtWidgets.QLabel("Move to Channel:")
        self.move_to_channel = QtWidgets.QLineEdit()
        self.move_to_channel.setToolTip("This is the channel that you will move selected cells too.")
        self.move_to_channel.setText(default_move_channel)
        self.move_to_channel.textChanged.connect(lambda: moveToChannel(self, 'main'))

        move_to_channel_layout = QtWidgets.QHBoxLayout()
        move_to_channel_layout.addWidget(move_to_channel_label)
        move_to_channel_layout.addWidget(self.move_to_channel)

        # -------- spike parameter options -------------------------------------

        feature_options = ["None"] + list(feature_name_map.keys())

        spike_parameter_layout = QtWidgets.QHBoxLayout()
        x_axis_label = QtWidgets.QLabel("X-Axis")
        self.x_axis_cb = QtWidgets.QComboBox()
        for option in feature_options:
            self.x_axis_cb.addItem(option)
        self.x_axis_cb.setCurrentIndex(self.x_axis_cb.findText(defaultXAxis))

        self.x_axis_cb.setToolTip("Choose a feature to plot on the X-Axis")

        x_axis_layout = QtWidgets.QHBoxLayout()
        x_axis_layout.addWidget(x_axis_label)
        x_axis_layout.addWidget(self.x_axis_cb)

        y_axis_label = QtWidgets.QLabel("Y-Axis")
        self.y_axis_cb = QtWidgets.QComboBox()
        self.y_axis_cb.setToolTip("Choose a feature to plot on the Y-Axis")
        for option in feature_options:
            self.y_axis_cb.addItem(option)
        self.y_axis_cb.setCurrentIndex(self.y_axis_cb.findText(defaultYAxis))

        y_axis_layout = QtWidgets.QHBoxLayout()
        y_axis_layout.addWidget(y_axis_label)
        y_axis_layout.addWidget(self.y_axis_cb)

        z_axis_label = QtWidgets.QLabel("Z-Axis")
        self.z_axis_cb = QtWidgets.QComboBox()
        self.z_axis_cb.setToolTip("Choose a feature to plot on the Z-Axis")
        for option in feature_options:
            self.z_axis_cb.addItem(option)
        self.z_axis_cb.setCurrentIndex(self.z_axis_cb.findText(defaultZAxis))

        z_axis_layout = QtWidgets.QHBoxLayout()
        z_axis_layout.addWidget(z_axis_label)
        z_axis_layout.addWidget(self.z_axis_cb)

        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addLayout(x_axis_layout)
        axis_layout.addLayout(y_axis_layout)
        axis_layout.addLayout(z_axis_layout)

        tetrode_layout = QtWidgets.QHBoxLayout()
        tetrode_label = QtWidgets.QLabel("Tetrode:")
        self.tetrode_cb = QtWidgets.QComboBox()
        self.tetrode_cb.currentIndexChanged.connect(self.tetrode_changed)
        tetrode_layout.addWidget(tetrode_label)
        tetrode_layout.addWidget(self.tetrode_cb)

        spike_parameter_widgets = [tetrode_layout, axis_layout, move_to_channel_layout]

        spike_parameter_layout.addStretch(1)
        for i, widget in enumerate(spike_parameter_widgets):
            if 'Layout' in widget.__str__():
                spike_parameter_layout.addLayout(widget)
                spike_parameter_layout.addStretch(1)
            else:
                spike_parameter_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                spike_parameter_layout.addStretch(1)

        if openGL:
            self.feature_win = pg.GraphicsWindow()
            feature_win_layout = QtWidgets.QGridLayout()
            self.feature_win.setLayout(feature_win_layout)
            self.glViewWidget = gl.GLViewWidget()
            self.glViewWidget.setBackgroundColor('k')

            feature_win_layout.addWidget(self.glViewWidget)
        else:
            # self.feature_win = PltWidget(self)
            self.feature_win = MatplotlibWidget()
            self.feature_win.toolbar.hide()  # hide the toolbar

        self.unit_win = pg.GraphicsWindow()

        plot_layout = QtWidgets.QHBoxLayout()
        for _object in [self.feature_win, self.unit_win]:
            plot_layout.addWidget(_object)

        # --------- Create the Buttons at the bottom of the Main Window ------------- #

        button_layout = QtWidgets.QHBoxLayout()

        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.plot_btn.clicked.connect(lambda: manage_features(self))

        self.reload_cut_btn = QtWidgets.QPushButton("Reload Cut")
        self.reload_cut_btn.clicked.connect(self.reload_cut)

        self.save_btn = QtWidgets.QPushButton("Save Cut")
        self.save_btn.clicked.connect(self.save_function)
        self.save_btn.setToolTip('Save to the current cut file')

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(lambda: undo_function(self))
        self.undo_btn.setToolTip('Click to undo previous action!')

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close_app)
        self.quit_btn.setShortcut("Ctrl+Q")  # creates shortcut for the quit button
        self.quit_btn.setToolTip('Click to quit gebaSpike!')

        button_order = [self.plot_btn, self.save_btn, self.undo_btn, self.reload_cut_btn, self.quit_btn]

        for btn in button_order:
            button_layout.addWidget(btn)

        main_window_layout = QtWidgets.QVBoxLayout()

        layout_order = [filename_grid_layout, spike_parameter_layout, plot_layout, button_layout]
        add_Stretch = [False, False, False, False]
        # ---------------- add all the layouts and widgets to the Main Window's layout ------------ #

        # main_window_layout.addStretch(1)  # adds the widgets/layouts according to the order
        for widget, addStretch in zip(layout_order, add_Stretch):
            if 'Layout' in widget.__str__():
                main_window_layout.addLayout(widget)
                if addStretch:
                    main_window_layout.addStretch(1)
            else:
                main_window_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
                if addStretch:
                    main_window_layout.addStretch(1)

        self.setLayout(main_window_layout)  # defining the layout of the Main Window

        self.show()  # shows the window

    def save_function(self):

        if self.cut_filename.text() == default_filename:
            return

        save_filename = os.path.realpath(self.cut_filename.text())

        if os.path.exists(save_filename):
            self.choice = None
            self.LogError.signal.emit('OverwriteCut!%s' % save_filename)
            while self.choice is None:
                time.sleep(0.1)

            if self.choice != QtWidgets.QMessageBox.Yes:
                return

        if len(self.tetrode_data) == 0:
            return

        # organize the cut data
        n_spikes = self.tetrode_data.shape[1]
        cut_values = np.zeros((n_spikes))

        for cell, cell_indices in self.cell_indices.items():
            cut_values[cell_indices] = cell

        # save the cut filename
        write_cut(save_filename, cut_values)

    def close_app(self):
        """This method will prompt the user, asking if they would like to quit or not"""
        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtWidgets.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def raiseError(self, error):

        if 'TetrodeExistError' in error:

            filename = error.split('!')[1]

            self.choice = QtWidgets.QMessageBox.question(self, "Tetrode Filename Does Not Exist!",
                                                         "The following tetrode filename does not exist: \n%s\nPlease" %
                                                         filename +
                                                         " ensure that the file exists before attempting to plot the"
                                                         " data",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidSession' in error:

            session = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Session Filename!",
                                                         "The following session filename is invalid: \n%s\nPlease" %
                                                         session +
                                                         " ensure that the appropriate files exist for this session.",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidCut' in error:

            session = error.split('!')[1]
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Cut/Clu Filename!",
                                                         "The following Cut filename is invalid: \n%s\nPlease" %
                                                         session +
                                                         " ensure that this .cut/clu filename belongs to this session!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'ChooseSession' in error:

            self.choice = QtWidgets.QMessageBox.question(self, "Choose Session Filename!",
                                                         "You have not chosen a session file yet! Please choose a"
                                                         " session filename before proceeding!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'InvalidMoveChannel' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Move to Channel Value!",
                                                         "The value you have chosen for the 'Move to Channel' value is "
                                                         "invalid, please choose a valid value before continuing!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'SameChannelInvalid' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Same Channel Error!",
                                                         "The value you have chosen for the 'Move to Channel' value is "
                                                         "the same as the cell you are cutting from! If you would like "
                                                         "to move these selected spikes to a different channel, please "
                                                         "choose another channel!",
                                                         QtWidgets.QMessageBox.Ok)

        elif 'ActionsMade' in error:
            self.choice = QtWidgets.QMessageBox.question(self, "Are you sure............?",
                                                         "You have performed some actions that will be lost when you"
                                                         " reload this cut file. Are you sure you want to continue?",
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Yes)

        elif 'OverwriteCut' in error:
            cut_file = os.path.realpath(error.split('!')[1])
            self.choice = QtWidgets.QMessageBox.question(self, "Cut Filename Exists",
                                                         "The following cut filename exists:\n\n%s\n\n Are you sure you " 
                                                         "want to overwrite this file?" % cut_file,
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def get_settings(self):
        settings = {}
        if os.path.exists(self.settings_filename):
            with open(self.settings_filename, 'r') as f:
               settings = json.load(f)
        return settings

    def overwrite_settings(self):
        # overwrite settings file
        with open(self.settings_filename, 'w') as f:
            json.dump(self.settings, f, sort_keys=True, indent=4)

    def reset_parameters(self):
        self.feature_data = None
        self.tetrode_data = None
        self.cut_data = None
        self.cut_data_original = None
        self.n_channels = None
        self.spike_times = None
        self.scatterItem = None
        self.cell_indices = {}
        self.plot_lines = {}
        self.avg_plot_lines = {}
        self.vb = {}
        self.active_ROI = []
        self.unit_data = {}
        self.cell_subsample_i = {}
        self.unit_positions = {}
        self.original_cell_count = {}

        self.latest_actions = {}

        self.actions_made = False

        self.last_drag_index = None

        self.drag_active = False

        self.unit_drag_lines = {}

        self.samples_per_spike = None
        self.spike_colors = None

        self.unit_plots = {}
        self.unit_rows = 0
        self.unit_cols = 0

        self.reset_plots()

    def reset_plots(self):

        self.feature_win.clear()
        self.unit_win.clear()

    def choose_cut_filename(self):
        if 'file_directory' not in self.settings.keys():
            current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        else:
            if os.path.exists(self.settings['file_directory']):
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Cut/.Clu' file!", directory=self.settings['file_directory'],
                    filter='Cut Files (*.cut, *.clu*)')
            else:
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Cut/.Clu' file!", directory='', filter='Cut Files (*.cut, *.clu*)')

        # if no file chosen, skip
        if current_filename == '':
            return

        chosen_directory = os.path.dirname(current_filename)

        self.settings['file_directory'] = chosen_directory

        self.overwrite_settings()

        cut_valid, error_raised = validate_cut(self, self.filename.text(), current_filename)

        if cut_valid:
            # replace the current .cut field in the choose .cut field with chosen filename
            self.cut_filename.setText(current_filename)

        else:
            if not error_raised:
                self.choice = None
                self.LogError.signal.emit('InvalidCut!%s' % current_filename)
                while self.choice is None:
                    time.sleep(0.1)

    def reload_cut(self):

        if self.actions_made is True:
            self.choice = None
            self.LogError.signal.emit('ActionsMade')
            while self.choice is None:
                time.sleep(0.1)

        if self.choice == QtWidgets.QMessageBox.Yes:
            self.reset_plots()
            self.reset_parameters()

            manage_features(self)

    def choose_filename(self):
        """
        This method will allow you to choose a filename to analyze.
        :return:
        """

        if 'file_directory' not in self.settings.keys():
            current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        else:
            if os.path.exists(self.settings['file_directory']):
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Set' file!", directory=self.settings['file_directory'],
                    filter='Set Files (*.set)')
            else:
                current_filename, filename_filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption="Select a '.Set' file!", directory='', filter='Set Files (*.set)')

        # if no file chosen, skip
        if current_filename == '':
            return

        chosen_directory = os.path.dirname(current_filename)

        self.settings['file_directory'] = chosen_directory

        self.overwrite_settings()

        session_valid, error_raised = validate_session(self, current_filename)

        if session_valid:
            # replace the current .set field in the choose .set field with chosen filename
            self.filename.setText(current_filename)

            self.reset_parameters()

        else:
            if not error_raised:
                self.choice = None
                self.LogError.signal.emit('InvalidSession!%s' % current_filename)
                while self.choice is None:
                    time.sleep(0.1)

    def set_cut_filename(self):
        filename = self.filename.text()
        tetrode = int(self.tetrode_cb.currentText())
        cut_filename = '%s_%d.cut' % (os.path.splitext(filename)[0], tetrode)
        self.cut_filename.setText(cut_filename)

    def tetrode_changed(self):

        # we will update the cut_filename
        self.set_cut_filename()

    def cut_filename_changed(self):
        """
        This method will run when the cut filename LineEdit has been changed
        """
        cut_filename = self.cut_filename.text()
        return

    def filename_changed(self):
        """
        This method will run when the filename LineEdit has been changed.

        It will essentially find the active tetrodes and populate the drop-down menu.
        """

        filename = self.filename.text()
        if os.path.exists(filename):
            self.tetrode_cb.clear()

            tetrode_list = find_tetrodes(self.filename.text())

            for file in tetrode_list:
                tetrode = os.path.splitext(file)[-1][1:]
                self.tetrode_cb.addItem(tetrode)

        self.set_cut_filename()


def launch_gui():
    """
    This function will launch the gebaSpike GUI
    :return:
    """

    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()  # Creating the Main Window
    main_window.raise_()  # raises the Main Window

    sys.exit(app.exec_())  # prevents the window from immediately exiting out


if __name__ == '__main__':
    launch_gui()
