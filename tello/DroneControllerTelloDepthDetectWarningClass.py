#from djitellopy import tello
#import logging
#import KeybordController as kc
from time import sleep
import cv2
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
import numpy as np
import pickle
captureImage = False
from TelloDroneUtils import TelloDroneUtils
##############Parameters##############


##############Drone init##############
#tello.Tello.setLevel(logging.WARNING)






custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
model_file = "modelOK.h5"
model = load_model(model_file, custom_objects=custom_objects, compile=False)
model.summary()
#x = 0 / 0
print('\nModel loaded ({0}).'.format(model_file))

tdu = TelloDroneUtils(model, False)

while not(tdu.end_program):
    # motion control
    vals = tdu.getKeyboardInput()
    tdu.me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    text = ["Battery: " + str(tdu.me.get_battery()) + "%",
            "Yaw: " + str(tdu.me.get_yaw()),
            "Pitch: " + str(tdu.me.get_pitch()),
            "Roll: " + str(tdu.me.get_roll()),
            "Height: " + str(tdu.me.get_height())]

    """
    Acc x: " + str(tdu.me.get_acceleration_x()),
    "Acc y: " + str(tdu.me.get_acceleration_y()),
    "Acc z: " + str(tdu.me.get_acceleration_z()),
    """

    tdu.kc.drawWarning(tdu.warning, text)
    tdu.end_program = vals[4]
    sleep(0.001)


tdu.endConnection()
cv2.destroyAllWindows()
