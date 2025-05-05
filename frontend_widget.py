import multiprocessing
import numpy as np
from pyqtgraph import ColorMap
from PyQt5 import QtWidgets
import pyqtgraph as pg
from configuration import config
from create_walking_path import get_treadmill_information
import cv2
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

# Configuration
rows = config['ROWS']
cols = config['COLS']

# 🛑 No more global queues here
mats_for_treadmill = get_treadmill_information()

# Create treadmill matrix dimensions
treadmill_rows = rows
treadmill_cols = cols * len(mats_for_treadmill)

# Create z_data_last for single and multiple mats
z_data_last_single = np.zeros((rows, cols))
z_data_last_treadmill = np.zeros((treadmill_rows, treadmill_cols))

class PressureMapAppSingle(QtWidgets.QMainWindow):
    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue  # 🆕 Save queue
        self.initUI()

    def initUI(self):
        self.setWindowTitle(f'{rows}x{cols} - Data Visualization')
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        layout.addWidget(self.image_view)

        colors = [
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]
        cmap = ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.image_view.setColorMap(cmap)

        self.timer = pg.QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_single_heatmap)
        self.timer.start(5)

        self.show()

    def update_single_heatmap(self):
        global z_data_last_single
        try:
            z_data = self.data_queue.get_nowait()  # 🆕 use self.data_queue
            z_data_last_single = z_data
        except:
            z_data = z_data_last_single

        z_data = np.array(z_data)
        high_res_matrix = zoom(z_data, 1, order=3)
        high_res_matrix_smooth = gaussian_filter(high_res_matrix, sigma=0.35)
        self.image_view.setImage(high_res_matrix_smooth, autoLevels=True)

class PressureMapAppTreadmill(QtWidgets.QMainWindow):
    def __init__(self, visual_queue):
        super().__init__()
        self.visual_queue = visual_queue  # 🆕 Save queue
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Visualization')
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        colors = [
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]
        cmap = ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.image_view.setColorMap(cmap)

        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()

        self.timer = pg.QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_treadmill_heatmap)
        self.timer.start(5)

        self.show()

    def update_treadmill_heatmap(self):
        global z_data_last_treadmill
        try:
            z_data = self.visual_queue.get_nowait()  # 🆕 use self.visual_queue
            z_data_last_treadmill = z_data
            while not self.visual_queue.empty():
                self.visual_queue.get()
        except:
            z_data = z_data_last_treadmill

        z_data = np.array(z_data)
        self.image_view.setImage(z_data, autoLevels=True)
