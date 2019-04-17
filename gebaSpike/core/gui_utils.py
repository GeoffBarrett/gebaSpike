from PyQt5 import QtWidgets, QtCore
from core.default_parameters import default_filename
import time
import os

Large_Font = ("Verdana", 12)  # defines two fonts for different purposes (might not be used
Small_Font = ("Verdana", 8)


def center(self):
    """
    centers the window on the screen
    """
    frameGm = self.frameGeometry()
    screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
    centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


class Worker(QtCore.QObject):
    """
    This Worker class is for threading using PyQt
    """
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)


class Communicate(QtCore.QObject):
    """
    A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window
    """
    signal = QtCore.pyqtSignal(str)


def validate_session(self, set_filename):

    if set_filename == default_filename:
        self.choice = None
        self.LogError.signal.emit('ChooseSession')
        while self.choice is None:
            time.sleep(0.1)
        return False, True
    return True, False


def validate_cut(self, set_filename, cut_filename):
    set_basename = os.path.basename(set_filename)

    if set_basename in cut_filename:
        return True
    else:
        return False
    return True
