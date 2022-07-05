import os
import glob
from datetime import datetime
import cv2
from KeybordControllerClass import KeybordControllerClass
from djitellopy import tello
import logging
import pickle
import numpy as np
from time import sleep
from utils import predict, load_images, display_images

from threading import Thread

def task_rgb(my_self):
    #global my_self.frame_to_process, my_self.task_lock, captureImage
    while not (my_self.end_program):
        img = my_self.me.get_frame_read().frame

        if my_self.captureImage:
            my_self.captureImage = False
            my_self.saveImage(img, True)

        if my_self.if_save_video:
            my_self.saveVideo(img)

        dst = cv2.undistort(img, my_self.mtx, my_self.dist, None, my_self.newcameramtx)
        x, y, w, h = my_self.roi
        dst = dst[y:y + h, x:x + w]
        # cv2.imshow('calibresult.png', dst)

        # frame = np.copy(img)
        # frame = cv2.resize(frame, (320, 320))
        # if not task_lock:
        #    frame_to_process = cv2.resize(frame, (320, 240))
        if not my_self.task_lock:
            frame = np.copy(dst)
            frame = cv2.resize(frame, (640, 480))
            my_self.frame_to_process = frame

        cv2.imshow("color", dst)


        # cv2.imshow("depth", im_color)
        cv2.waitKey(1)
        # sleep(0.01)


def task_depth(my_self):
    # global task_lock, captureImage
    while not (my_self.end_program):
        if my_self.frame_to_process is not None:
            my_self.task_lock = True
            x = np.clip(my_self.frame_to_process / 255, 0, 1)
            inputs = np.expand_dims(x, 0)
            outputs = predict(my_self.model, inputs)
            my_self.task_lock = False
            # RGB and depth output
            img = np.copy(outputs[0, :, :, :])

            ####################################################################################
            # size = 10

            # img = img / np.max(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            img = (255 * img).astype(np.uint8)

            """
            cluster_scale = 4
            eps = 5
            min_sample = 10
            how_many_percent_first_take = 0.5
            """
            depth = cv2.resize(img, (int(img.shape[0] / my_self.cluster_scale),
                                     int(img.shape[1] / my_self.cluster_scale)),
                               interpolation=cv2.INTER_LINEAR)

            # depth = result = ndimage.median_filter(depth, size=5)

            XX = np.zeros((depth.shape[0] * depth.shape[1], 3))
            id_help = 0
            for a in range(depth.shape[0]):
                for b in range(depth.shape[1]):
                    XX[id_help, 0] = my_self.cluster_scale * a / 2
                    XX[id_help, 1] = my_self.cluster_scale * b / 2
                    XX[id_help, 2] = depth[a, b]
                    if XX[id_help, 2] > my_self.depth_threshold:
                        XX[id_help, 2] = np.max(depth)
                    id_help = id_help + 1

            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=my_self.eps, min_samples=my_self.min_sample).fit(XX)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if my_self.verbose:
                print("Estimated number of clusters: %d" % n_clusters_)
                print("Estimated number of noise points: %d" % n_noise_)

            wwww = np.zeros((depth.shape[0], depth.shape[1], 3))
            wwww2 = np.zeros((depth.shape[0], depth.shape[1]))

            a_set = set(db.labels_)
            a_set_id = list(a_set)
            number_of_unique_values = len(a_set)
            eleCount = []
            eleAverage = []
            labelslist = db.labels_.tolist()
            #id_count = 0
            for ids in a_set:
                eleCount.append(labelslist.count(ids))
                eleAverage.append(0)
            for x in range(int(wwww.shape[0])):
                for y in range(int(wwww.shape[1])):
                    if (depth[x, y] < my_self.depth_threshold):
                        label = db.labels_[y + x * wwww.shape[1]]
                        id = a_set_id.index(label)
                        eleAverage[id] = eleAverage[id] + depth[x, y]
            for id in range(len(eleCount)):
                eleAverage[id] = eleAverage[id] / eleCount[id]

            eleAverageSorted = np.argsort(eleAverage).tolist()
            # how_many_percent_first_take = 0.75
            how_many_first_take = int(len(eleAverageSorted) * my_self.how_many_percent_first_take)
            for x in range(int(wwww.shape[0])):
                for y in range(int(wwww.shape[1])):
                    if (depth[x, y] < my_self.depth_threshold):
                        label = db.labels_[y + x * wwww.shape[1]]

                        id = a_set_id.index(label)
                        ele_sorted_id = eleAverageSorted.index(id)
                        if ele_sorted_id < how_many_first_take and eleAverage[id] < my_self.depth_threshold2:
                            # id = a_set_id.index(label)

                            # if eleAverage[id] < 128:
                            wwww[x, y, 0] = 255  # eleAverage[id]
                            wwww[x, y, 1] = eleAverage[id]
                            wwww[x, y, 2] = eleAverage[id]

            wwww = wwww.astype(np.uint8)

            ############################################

            img_width = wwww.shape[1]
            img_height = wwww.shape[0]

            # print(str(img_width) + " " + str(img_height))
            img_width_half = int(img_width / 2)
            img_height_half = int(img_height / 2)

            mean_value = np.mean(wwww[(img_height_half - my_self.warning_size + my_self.warning_offset):(img_height_half + my_self.warning_size + my_self.warning_offset),
                                 (img_width_half - my_self.warning_size):(img_width_half + my_self.warning_size),
                                 0]) / 255

            my_color = (0, 0, 255)
            if mean_value > my_self.warning_threshold:
                my_self.warning = True
            else:
                my_self.warning = False
                my_color = (0, 255, 0)
                #print("warning!!!!")


            cv2.rectangle(wwww, (img_width_half - my_self.warning_size, img_height_half - my_self.warning_size + my_self.warning_offset),
                          (img_width_half + my_self.warning_size, img_height_half + my_self.warning_size + my_self.warning_offset), color=my_color)
            ############################################

            wwww = cv2.resize(wwww, (640, 480))
            cv2.imshow("warning", wwww)

            im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            im_color = cv2.resize(im_color, (640, 480))
            cv2.imshow("a", im_color)

            cv2.waitKey(1)
        sleep(0.01)

class TelloDroneUtils:
    def __init__(self, model, verbose=True, initTello=True, videoFromDir=None, videoFileName=None):
        self.end_program = False
        if initTello:
            self.kc = KeybordControllerClass()

        self.me = None

        if initTello:
            self.me = tello.Tello()
            self.me.LOGGER.setLevel(logging.WARNING)
            self.me.connect()
        #print(self.me.get_battery())

        self.speed = 50
        self.captureImage = False
        self.frame_to_process = None
        self.task_lock = False

        self.depth_threshold = 256
        self.depth_threshold2 = 96
        self.warning_threshold = 0.3
        self.warning_size = 10
        self.warning_offset = -self.warning_size * 2

        self.cluster_scale = 4
        self.eps = 5
        self.min_sample = 10
        self.how_many_percent_first_take = 0.5

        self.verbose = verbose
        self.warning = False

        my_pickle = open("calibration_parameters.p", "rb")
        [self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.newcameramtx, self.roi, self.mapx, self.mapy] = pickle.load(my_pickle)

        self.if_save_video = False

        if initTello:
            [dir_name, file_name] = self.createDirToSaveVideo()

        self.model = model

        if initTello:
            self.me.streamon()

            self.t1 = Thread(target=task_rgb, args=(self,))
            self.t1.start()

            self.t2 = Thread(target=task_depth, args=(self,))
            self.t2.start()

        if not(initTello) and videoFromDir is not None and videoFileName is not None :
            self.task_generate_video(videoFromDir, videoFileName)

    def endConnection(self):
        self.t1.join(timeout=1)
        self.t2.join(timeout=1)
        #self.me.streamoff()
        #self.me.end()

    def getKeyboardInput(self):
        #global captureImage
        # lr - left / right
        # fb - forward / backward
        # ud - up / down
        # yv - yaw velocity

        end_program = False

        lr, fb, ud, yv = 0, 0, 0, 0
        if self.kc.getKey('LEFT'):
            lr = -self.speed
        elif self.kc.getKey('RIGHT'):
            lr = self.speed

        if self.kc.getKey('UP'):
            fb = self.speed
        elif self.kc.getKey('DOWN'):
            fb = -self.speed

        if self.kc.getKey('w'):
            ud = self.speed
        elif self.kc.getKey('s'):
            ud = -self.speed

        if self.kc.getKey('a'):
            yv = -self.speed
        elif self.kc.getKey('d'):
            yv = self.speed

        if self.kc.getKey('p'): self.captureImage = True

        if self.kc.getKey("1"): self.me.takeoff()
        if self.kc.getKey("2"): self.me.land()

        if self.kc.getKey("ESCAPE"): end_program = True

        return [lr, fb, ud, yv, end_program]

    def createDirToSaveVideo(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H-%M-%S.%f")
        self.dir_name = "./video/" + dt_string
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        self.file_name = self.dir_name + "/description.txt"
        return [self.dir_name, self.file_name]

    def saveVideo(self, img):
        hs = open(self.file_name, "a")
        now = datetime.now()
        date_str = str(now.now())
        dt_string = now.strftime("%d-%m-%Y %H-%M-%S.%f") + ".png"
        hs.write(date_str + "," + dt_string + "\n")
        cv2.imwrite(self.dir_name + "/" + dt_string, img)
        hs.close()

    def saveImage(self, img):
        hs = open(self.file_name, "a")
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H-%M-%S.%f") + ".png"
        cv2.imwrite("./video/" + dt_string, img)
        if self.verbose:
            print("Image saved as " + dt_string)
        hs.close()

    def task_generate_video(self, videoFromDir, videoFileName):
        fps = 30
        #out = cv2.VideoWriter(videoFileName, cv2.VideoWriter_fourcc(*'MP42'), fps, (1024, 576), False)
        #out = cv2.VideoWriter(videoFileName + ".avi", cv2.VideoWriter_fourcc(*'x264'), fps, (1024, 576), False)
        out = cv2.VideoWriter(videoFileName, cv2.VideoWriter_fourcc(*'x264'), fps, (1024, 576), True)
        #out = cv2.VideoWriter(videoFileName + ".avi", cv2.VideoWriter_fourcc(*'x264'), fps, (640, 480), True)

        list_of_files = sorted(filter(os.path.isfile,
                                      #glob.glob(videoFromDir + '/*.png')))
                                      glob.glob(videoFromDir + '*.png')))
        # Iterate over sorted list of files and print the file paths
        # one by one.
        #for a in range(30):
        #    file_path = list_of_files[a]
        for file_path in list_of_files:
            img = cv2.imread(file_path)

            dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
            x, y, w, h = self.roi
            dst = dst[y:y + h, x:x + w]
            #cv2.imshow("color", dst)
            frame_to_process = np.copy(dst)
            frame_to_process = cv2.resize(frame_to_process, (640, 480))
            x = np.clip(frame_to_process / 255, 0, 1)
            inputs = np.expand_dims(x, 0)
            outputs = predict(self.model, inputs)
            # RGB and depth output
            img = np.copy(outputs[0, :, :, :])

            ####################################################################################
            # size = 10

            # img = img / np.max(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            img = (255 * img).astype(np.uint8)


            depth = cv2.resize(img, (int(img.shape[0] / self.cluster_scale),
                                     int(img.shape[1] / self.cluster_scale)),
                               interpolation=cv2.INTER_LINEAR)

            # depth = result = ndimage.median_filter(depth, size=5)

            XX = np.zeros((depth.shape[0] * depth.shape[1], 3))
            id_help = 0
            """
            for a in range(depth.shape[0]):
                for b in range(depth.shape[1]):
                    XX[id_help, 0] = self.cluster_scale * a / 2
                    XX[id_help, 1] = self.cluster_scale * b / 2
                    XX[id_help, 2] = depth[a, b]
                    if XX[id_help, 2] > self.depth_threshold:
                        XX[id_help, 2] = np.max(depth)
                    id_help = id_help + 1
            """
            XX0 = [i for i in range(depth.shape[1]) for j in range(depth.shape[0])]
            XX1 = [j for i in range(depth.shape[0]) for j in range(depth.shape[1])]
            XX[:, 0] = XX0
            XX[:, 1] = XX1
            XX[:, 0] = XX[:, 0] * self.cluster_scale / 2
            XX[:, 1] = XX[:, 1] * self.cluster_scale / 2
            XX2 = [depth[a, b] for a in range(depth.shape[0]) for b in range(depth.shape[1])]
            XX[:, 2] = XX2


            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(XX)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if self.verbose:
                print("Estimated number of clusters: %d" % n_clusters_)
                print("Estimated number of noise points: %d" % n_noise_)

            wwww = np.zeros((depth.shape[0], depth.shape[1], 3))
            wwww2 = np.zeros((depth.shape[0], depth.shape[1]))

            a_set = set(db.labels_)
            a_set_id = list(a_set)
            number_of_unique_values = len(a_set)
            eleCount = []
            eleAverage = []
            labelslist = db.labels_.tolist()
            #id_count = 0
            for ids in a_set:
                eleCount.append(labelslist.count(ids))
                eleAverage.append(0)
            for x in range(int(wwww.shape[0])):
                for y in range(int(wwww.shape[1])):
                    if (depth[x, y] < self.depth_threshold):
                        label = db.labels_[y + x * wwww.shape[1]]
                        id = a_set_id.index(label)
                        eleAverage[id] = eleAverage[id] + depth[x, y]
            for id in range(len(eleCount)):
                eleAverage[id] = eleAverage[id] / eleCount[id]

            eleAverageSorted = np.argsort(eleAverage).tolist()
            # how_many_percent_first_take = 0.75
            how_many_first_take = int(len(eleAverageSorted) * self.how_many_percent_first_take)
            for x in range(int(wwww.shape[0])):
                for y in range(int(wwww.shape[1])):
                    if (depth[x, y] < self.depth_threshold):
                        label = db.labels_[y + x * wwww.shape[1]]

                        id = a_set_id.index(label)
                        ele_sorted_id = eleAverageSorted.index(id)
                        if ele_sorted_id < how_many_first_take and eleAverage[id] < self.depth_threshold2:
                            # id = a_set_id.index(label)

                            # if eleAverage[id] < 128:
                            wwww[x, y, 0] = 255  # eleAverage[id]
                            wwww[x, y, 1] = eleAverage[id]
                            wwww[x, y, 2] = eleAverage[id]

            wwww = wwww.astype(np.uint8)

            ############################################

            img_width = wwww.shape[1]
            img_height = wwww.shape[0]

            # print(str(img_width) + " " + str(img_height))
            img_width_half = int(img_width / 2)
            img_height_half = int(img_height / 2)

            mean_value = np.mean(wwww[(img_height_half - self.warning_size + self.warning_offset):(img_height_half + self.warning_size + self.warning_offset),
                                 (img_width_half - self.warning_size):(img_width_half + self.warning_size),
                                 0]) / 255

            my_color = (0, 0, 255)
            if mean_value > self.warning_threshold:
                self.warning = True
            else:
                self.warning = False
                my_color = (0, 255, 0)
                #print("warning!!!!")


            cv2.rectangle(wwww, (img_width_half - self.warning_size, img_height_half - self.warning_size + self.warning_offset),
                          (img_width_half + self.warning_size, img_height_half + self.warning_size + self.warning_offset), color=my_color)
            ############################################

            wwww = cv2.resize(wwww, (640, 480))
            #cv2.imshow("warning", wwww)

            im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            im_color = cv2.resize(im_color, (640, 480))
            #cv2.imshow("a", im_color)


            ret_image = np.zeros((576, 1024, 3))
            part1 = cv2.resize(dst, (640, 576))
            ret_image[0:576, 0:640, :] = part1[:,:,:]
            part2 = cv2.resize(im_color, (384, 288))
            part3 = cv2.resize(wwww, (384, 288))

            ret_image[0:288, 640:1024, :] = part2[:, :, :]
            ret_image[288:576, 640:1024, :] = part3[:, :, :]

            ret_image = ret_image.astype(np.uint8)
            # cv2.imshow("depth", im_color)
            cv2.imshow("output", ret_image)
            """
            ret_image = np.zeros((480, 640, 3))
            part1 = cv2.resize(dst, (480, 480))
            ret_image[0:480, 0:480, :] = part1[:, :, :]
            part2 = cv2.resize(im_color, (160, 240))
            part3 = cv2.resize(wwww, (160, 240))

            ret_image[0:240, 480:640, :] = part2[:, :, :]
            ret_image[240:480, 480:640, :] = part3[:, :, :]

            ret_image = ret_image.astype(np.uint8)
            # cv2.imshow("depth", im_color)
            cv2.imshow("output", ret_image)
            """
            cv2.waitKey(1)
            out.write(ret_image)

        out.release()