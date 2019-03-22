import os
import time
from core.gui_utils import validate_session
from core.Tint_Matlab import find_unit


def plot_features(self):

    session_filename = self.filename.text()

    session_valid, error_raised = validate_session(self, session_filename)

    if session_valid:
        tetrode = self.tetrode_cb.currentText()

        session_filepath = os.path.splitext(self.filename.text())[0]

        tetrode_filename = '%s.%s' % (session_filepath, tetrode)

        if os.path.exists(tetrode_filename):
            cut_mat, cut_unique = find_unit(os.path.dirname(session_filepath), [tetrode_filename])

        else:
            # filename does not exist
            self.choice = None
            self.LogError.signal.emit('TetrodeExistError!%s' % tetrode_filename)

            while self.choice is None:
                time.sleep(0.1)

        return
