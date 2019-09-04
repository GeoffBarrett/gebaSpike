from PyQt5 import QtCore

project_name = 'gebaSpike'

default_filename = 'Choose a filename!'

defaultXAxis = 'PC1'
defaultYAxis = 'PC2'
defaultZAxis = 'PC3'

# openGL = False
openGL = True

gridLines = True

feature_spike_size = 2
feature_spike_opacity = 0.5

# unitMode = 'MatplotWidget'
unitMode = 'PyQt'

default_move_channel = str(0)

channel_range = 256

max_spike_plots = 2000

max_num_actions = 5  # max number of actions to remember

# setting the back quote button (shares with the tilde/~ button)
alt_action_button = QtCore.Qt.Key_QuoteLeft