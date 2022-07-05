from djitellopy import tello
import logging
import KeybordController as kc
from time import sleep
import cv2
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
import numpy as np
import pickle
import os
captureImage = False
from TelloDroneUtils import TelloDroneUtils
##############Parameters##############


##############Drone init##############
#tello.Tello.setLevel(logging.WARNING)

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
model_file = "modelOK.h5"
model = load_model(model_file, custom_objects=custom_objects, compile=False)
#model.summary()
print('\nModel loaded ({0}).'.format(model_file))


from glob import glob
"""
list_of_files = sorted(filter(os.path.isfile,
                              # glob.glob(videoFromDir + '/*.png')))
                              glob.glob(videoFromDir + '*.png')))

"""
dir_list = glob("d:/dane/2022-06-28/video/vid/*/", recursive = True)
for dd in dir_list:
    dir_name = os.path.basename(os.path.dirname(dd))
    #if int(dir_name) > 87:
    print(os.path.basename(os.path.dirname(dd)))
    dd = dd.replace("\\","/")
    tdu = TelloDroneUtils(model, False, initTello=False,
                          videoFromDir= dd,
                          videoFileName='d:/dane/2022-06-28/video/out_vid/' + dir_name + ".mp4")
    cv2.destroyAllWindows()
"""

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
model_file = "modelOK.h5"
model = load_model(model_file, custom_objects=custom_objects, compile=False)
#model.summary()
print('\nModel loaded ({0}).'.format(model_file))

tdu = TelloDroneUtils(model, False, initTello=False,
                      videoFromDir='d:/dane/2022-06-28/video/vid/000/',
                      videoFileName='d:/dane/2022-06-28/video/vid/000.mp4')
cv2.destroyAllWindows()
"""