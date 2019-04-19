import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from exporters import ImageExporter


def get_channel_color(cell_number):

    """This method will match the cell number with the color it should be RGB in Tint.
    These cells are numbered from 1-30 (there is technically a zeroth cell, but that isn't plotted"""
    spike_colors = [(1, 8, 184), (93, 249, 75), (234, 8, 9),
                    (229, 22, 239), (80, 205, 243), (27, 164, 0),
                    (251, 188, 56), (27, 143, 167), (127, 41, 116),
                    (191, 148, 23), (185, 9, 17), (231, 223, 67),
                    (144, 132, 145), (34, 236, 228), (217, 20, 145),
                    (172, 64, 80), (176, 106, 138), (199, 194, 167),
                    (216, 204, 105), (160, 204, 61), (187, 81, 88),
                    (45, 216, 122), (242, 136, 25), (50, 164, 161),
                    (249, 67, 16), (252, 232, 147), (114, 156, 238),
                    (241, 212, 179), (129, 62, 162), (235, 133, 126)]

    while cell_number > len(spike_colors)-1:
        cell_number = cell_number - len(spike_colors)

    return spike_colors[int(cell_number)-1]


class CustomViewBox(pg.ViewBox):
    """
    Subclass of ViewBox
    """

    def __init__(self, window, item, parent=None):
        """
        Constructor of the CustomViewBox
        """
        super(CustomViewBox, self).__init__(parent)
        # self.plot = plot
        self.window = window
        self.item = item
        self.menu = None  # Override pyqtgraph ViewBoxMenu
        self.menu = self.getMenu()  # Create the menu

    def raiseContextMenu(self, ev):
        """
        Raise the context menu
        """
        if not self.menuEnabled():
            return
        menu = self.getMenu()
        pos  = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.save_plot = QtWidgets.QAction("Save Figure", self.menu)
            self.save_plot.triggered.connect(self.export)
            self.menu.addAction(self.save_plot)
        return self.menu

    def export(self):
        # choose filename to save as
        save_filename = QtWidgets.QFileDialog.getSaveFileName(QtGui.QWidget(), 'Save Scores', '',
                                                          'PNG (*.png);;JPG (*.jpg);;TIF (*.tif);;GIF (*.gif)')

        if save_filename == '':
            return

        # create an exporter instance, as an argument give it
        # the item you wish to export

        if 'GraphicsWindow' in str(self.item):
            # get the main plot which occurs at row=1, and column=0
            plotitem = self.item.getItem(1, 0)
            # turn off the infinite line marking where the cursor is
            self.window.mouse_vLine.hide()

            exporter = ImageExporter(plotitem)

            # set export parameters if needed
            # exporter.parameters()['width'] = 100  # (note this also affects height parameter)

            # save to file
            exporter.export(save_filename)

            self.window.mouse_vLine.show()

        elif 'PltWidget' in str(self.item):
            plotitem = self.item.getPlotItem()

            exporter = ImageExporter(plotitem)

            # set export parameters if needed
            # exporter.parameters()['width'] = 100  # (note this also affects height parameter)

            # save to file
            exporter.export(save_filename)


class PltWidget(pg.PlotWidget):
    """
    Subclass of PlotWidget created so that we can have a custom viewBox with our own menu on right click
    """
    def __init__(self, window, parent=None):
        """
        Constructor of the widget
        """
        super(PltWidget, self).__init__(parent, viewBox=CustomViewBox(window, self))


class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self, x, y, *args, **kwargs):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        self._boundingRect = None

        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0  # don't draw the segment between each trace
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        pg.QtGui.QGraphicsPathItem.__init__(self, self.path)

        self.check_function_call = 0

        pen_kwargs = {}
        for kwarg in kwargs:
            if 'pen_' in kwarg:
                kwargNew = kwarg.split('pen_')[1]
                pen_kwargs[kwargNew] = kwargs[kwarg]

        if len(pen_kwargs) > 0:
            self.setPen(pg.mkPen(**pen_kwargs))
        else:
            self.setPen(pg.mkPen('w'))

    def shape(self):  # override because QGraphicsPathItem.shape is too expensive.
        return pg.QtGui.QGraphicsItem.shape(self)

    def boundingRect(self):
        if self._boundingRect is None:
            self._boundingRect = self.path.boundingRect()
        return self._boundingRect

