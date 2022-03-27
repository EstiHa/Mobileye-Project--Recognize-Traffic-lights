import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from tensorflow.keras.models import load_model

from part1.part1 import recognize_lights
# from part3.SFM_standAlone import run
from part3.SFM_standAlone import SFM_standAlone


class Controller:
    def __init__(self, pkl_path):
        self.__pkl_path = pkl_path
        self.__prev_frame = None
        self.__curr_frame = None
        self.__model = load_model("part2/model.h5")
        self.sfm = SFM_standAlone(pkl_path)

    def start(self, image):
        self.__prev_frame, self.__curr_frame = self.__curr_frame, image

        red_x, red_y, green_x, green_y = self.find_suspecious() #Run phase 1
        suspicious_images = self.crop_images(red_x, red_y, green_x, green_y) #יצירת תמונות קטנות סביב כל נקודה חשודה
        success_rate = self.classify_tfls(suspicious_images) #Run phase 2
        success_rate = np.argmax(success_rate, axis=-1)
        tfls = np.array(self.extract_tfls(red_x, red_y, green_x, green_y, success_rate))
        self.calc_distances(tfls) #Run phase 3

    def find_suspecious(self):
        return recognize_lights(self.__curr_frame)

    def add_border(self, image):
        old_im = Image.fromarray(image)
        old_size = old_im.size
        new_size = (old_size[0] + 82, old_size[1] + 82)
        new_im = Image.new("RGB", new_size)
        new_im.paste(old_im, ((new_size[0] - old_size[0]) // 2,
                              (new_size[1] - old_size[1]) // 2))
        return np.array(new_im)

    def crop_images(self, red_x, red_y, green_x, green_y):
        croped_images = []
        current_image_array = np.array(Image.open(self.__curr_frame))
        current_image_array = self.add_border(current_image_array)
        for y, x in zip(red_x, red_y):
            current_image = current_image_array[x:x+81, y :y+81, :]
            current_image = current_image.astype(np.uint8)  # changing the pictures from utf8 to float
            croped_images.append(current_image)
        for y, x in zip(green_x, green_y):
            current_image = current_image_array[x :x+81, y :y+81, :]
            current_image = current_image.astype(np.uint8)  # changing the pictures from utf8 to float
            croped_images.append(current_image)
        return croped_images

    def classify_tfls(self, suspicious_images):
        return self.__model.predict(np.asarray(suspicious_images))

    def extract_tfls(self, red_x, red_y, green_x, green_y, success_rate):
        x=red_x + green_x
        y=red_y + green_y
        tfls = []
        for i in range (len(success_rate)):
            if success_rate[i] == 1 :
                tfls.append([x[i], y[i]])
        return tfls

    def calc_distances(self, tfls):
        self.sfm.run(tfls, self.__curr_frame)
