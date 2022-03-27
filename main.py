from controller import Controller
from PIL import Image
import matplotlib.pyplot as plt


def read_image():
    images_path = []
    with open("play_list.pls") as f:
        all_row = f.readlines()
    controller = Controller(all_row[0][:-1])
    for i in range(2, len(all_row)):
        images_path.append(all_row[i][:-1])
    for path in images_path:
        print(path)
        controller.start(path)


if __name__ == '__main__':
    read_image()