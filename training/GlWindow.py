from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from GLWidget import GLWidget
import numpy as np
import time
import cv2
# Visualization
import matplotlib.pyplot as plt
plasma = plt.get_cmap('plasma')

from scipy import ndimage
from skimage.transform import resize
import os

from sklearn.cluster import DBSCAN

rgb_width = 640
rgb_height = 480

model_name = "gotowe/model_small.h5"

# Image shapes
height_rgb, width_rgb = 480, 640
height_depth, width_depth = height_rgb // 2, width_rgb // 2
rgb_width = width_rgb
rgb_height = height_rgb

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
global graph,model
graph = tf.compat.v1.get_default_graph()

class GlWindow(QtWidgets.QWidget):
    updateInput = QtCore.Signal()

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.model = None
        self.capture = None
        self.glWidget = GLWidget()

        mainLayout = QtWidgets.QVBoxLayout()

        # Input / output views
        viewsLayout = QtWidgets.QGridLayout()
        self.inputViewer = QtWidgets.QLabel("[Click to start]")
        self.inputViewer.setPixmap(QtGui.QPixmap(rgb_width ,rgb_height))
        self.outputViewer = QtWidgets.QLabel("[Click to start]")
        self.outputViewer.setPixmap(QtGui.QPixmap(rgb_width//2,rgb_height//2))

        imgsFrame = QtWidgets.QFrame()
        inputsLayout = QtWidgets.QVBoxLayout()
        imgsFrame.setLayout(inputsLayout)
        inputsLayout.addWidget(self.inputViewer)
        inputsLayout.addWidget(self.outputViewer)

        viewsLayout.addWidget(imgsFrame ,0 ,0)
        viewsLayout.addWidget(self.glWidget ,0 ,1)
        viewsLayout.setColumnStretch(1, 10)
        mainLayout.addLayout(viewsLayout)

        # Load depth estimation model
        toolsLayout = QtWidgets.QHBoxLayout()

        self.button = QtWidgets.QPushButton("Load model...")
        self.button.clicked.connect(self.loadModel)
        toolsLayout.addWidget(self.button)

        self.button5 = QtWidgets.QPushButton("Load image")
        self.button5.clicked.connect(self.loadImageFile)
        toolsLayout.addWidget(self.button5)

        self.button2 = QtWidgets.QPushButton("Webcam")
        self.button2.clicked.connect(self.loadCamera)
        toolsLayout.addWidget(self.button2)

        self.button3 = QtWidgets.QPushButton("Video")
        self.button3.clicked.connect(self.loadVideoFile)
        toolsLayout.addWidget(self.button3)

        self.button4 = QtWidgets.QPushButton("Pause")
        self.button4.clicked.connect(self.loadImage)
        toolsLayout.addWidget(self.button4)

        self.button6 = QtWidgets.QPushButton("Refresh")
        self.button6.clicked.connect(self.updateCloud)
        toolsLayout.addWidget(self.button6)

        mainLayout.addLayout(toolsLayout)

        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("RGBD Viewer"))

        # Signals
        self.updateInput.connect(self.update_input)

        # Default example
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        coloredDepth = (plasma(self.glWidget.depth[: ,: ,0])[: ,: ,:3] * 255).astype('uint8')
        self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))

    def loadModel(self):
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        tic()
        self.model = load_model()
        print('Model loaded.')
        toc()
        self.updateCloud()
        QtGui.QGuiApplication.restoreOverrideCursor()

    def loadCamera(self):
        self.capture = cv2.VideoCapture(0)
        self.updateInput.emit()

    def loadVideoFile(self):
        self.capture = cv2.VideoCapture('video.mp4')
        self.updateInput.emit()

    def loadImage(self):
        self.capture = None
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        self.updateCloud()

    def loadImageFile(self):
        self.capture = None
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select image', '', self.tr('Image files (*.jpg *.png)'))[0]
        img = QtGui.QImage(filename).scaledToHeight(rgb_height)
        xstart = 0
        if img.width() > rgb_width: xstart = (img.width() - rgb_width) // 2
        img = img.copy(xstart, 0, xstart +rgb_width, rgb_height)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(img))
        self.updateCloud()

    def update_input(self):
        # Don't update anymore if no capture device is set
        if self.capture == None: return

        # Capture a frame
        ret, frame = self.capture.read()

        # Loop video playback if current stream is video file
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()

        # Prepare image and show in UI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np_to_qimage(frame)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(image))

        # Update the point cloud
        self.updateCloud()

    def updateCloud(self):
        print("update cload")
        rgb8 = qimage_to_np(self.inputViewer.pixmap().toImage())
        """
        for x in range(rgb8.shape[0]):
            for y in range(rgb8.shape[1]):
                rgb8[x, y, 0] = 255
                rgb8[x, y, 1] = 0
        """


        """
        for x in range(rgb8.shape[0]):
            for y in range(rgb8.shape[1]):
                rgb8[x,y,0] = 255
                rgb8[x, y, 1] = rgb8[x, y, 2] = 0
        """
        self.glWidget.rgb = resize((rgb8[: ,: ,:3 ] /255)[: ,: ,::-1], (rgb_height, rgb_width), order=1, anti_aliasing=True)

        if self.model:
            with graph.as_default():
                depth = (1000 / self.model.predict( np.expand_dims(self.glWidget.rgb, axis=0)  )) / 1000
            coloredDepth = (plasma(depth[0 ,: ,: ,0])[: ,: ,:3] * 255).astype('uint8')


            X = depth[0 ,: ,: ,0]
            XX = np.zeros((int(X.shape[0] * X.shape[1] / 16), 3))
            #XX = np.random.rand(X.shape[0] * X.shape[1] / 4, 3)
            #print("YYY")


            #XX = np.zeros((X.shape[0] * X.shape[1],1))
            id_help = 0

            for a in range(int(X.shape[0] / 4)):
                for b in range(int(X.shape[1] / 4)):
                    #XX[id_help, 0] = a / X.shape[0]
                    #XX[id_help, 1] = b / X.shape[1]
                    XX[id_help, 0] = 2 * a
                    XX[id_help, 1] = 2 * b

                    XX[id_help, 2] = 255 * (X[int(4 * a), int(4 * b)] - np.min(X)) / (np.max(X) - np.min(X))
                    if XX[id_help, 2] > 128:
                        XX[id_help, 2] = 1000
                    id_help = id_help + 1



            #db = DBSCAN(eps=0.3, min_samples=10).fit(XX)
            import time
            start = time.time()
            db = DBSCAN(eps=5.0, min_samples=10).fit(XX)
            end = time.time()
            print(end - start)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)

            #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

            def getColor(val):
                if val % 6 == 0:
                    return [255, 0 ,0]
                if val % 6 == 1:
                    return [0, 255, 0]
                if val % 6 == 2:
                    return [0, 0, 255]
                if val % 6 == 3:
                    return [255, 255, 0]
                if val % 6 == 4:
                    return [0, 255, 255]
                if val % 6 == 5:
                    return [255, 0, 255]
                return [255,255,255]

            #count = 0
            for x in range(self.glWidget.rgb.shape[0]):
                for y in range(self.glWidget.rgb.shape[1]):
                    #ccc = getColor(db.labels_[int(x / 2) + int(y / 2) * int(self.glWidget.rgb.shape[0] / 2)])
                    ccc = getColor(db.labels_[int(y / 8) + int(x / 8) * int(self.glWidget.rgb.shape[1] / 8)])

                    self.glWidget.rgb[x, y, 0] = ccc[0]
                    self.glWidget.rgb[x, y, 1] = ccc[1]
                    self.glWidget.rgb[x, y, 2] = ccc[2]
                    #count = count + 1


            """
            for x in range(self.glWidget.rgb.shape[0]):
                for y in range(self.glWidget.rgb.shape[1]):
                    self.glWidget.rgb[x, y, 0] = 255
                    self.glWidget.rgb[x, y, 1] = rgb8[x, y, 2] = 0
            """

            """
            for x in range(coloredDepth.shape[0]):
                for y in range(coloredDepth.shape[1]):
                    coloredDepth[x, y, 0] = 255
                    coloredDepth[x, y, 1] = coloredDepth[x, y, 2] = 0
            """
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
            self.glWidget.depth = depth[0 ,: ,: ,0]
        else:
            self.glWidget.depth = 0.5 + np.zeros((rgb_height//2, rgb_width//2, 1))

        self.glWidget.updateRGBD()
        self.glWidget.updateGL()

        # Update to next frame if we are live
        QtCore.QTimer.singleShot(10, self.updateInput)


# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()


def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)


# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

# Function timing
ticTime = time.time()
def tic(): global ticTime; ticTime = time.time()
def toc(): print('{0} seconds.'.format(time.time() - ticTime))

def load_model():
    # Kerasa / TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    from keras.models import load_model
    from layers import BilinearUpSampling2D

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    # Load model into GPU / CPU
    return load_model(model_name, custom_objects=custom_objects, compile=False)