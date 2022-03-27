import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

from part3.SFM1 import prepare_3D_data, rotate, unnormalize, calc_TFL_dist


def visualize(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = rotate(norm_prev_pts, R)
    rot_pts = unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(unnormalize(np.array([norm_foe]), focal, pp))

    fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
    prev_sec.set_title('prev(' + str(prev_container.id) + ')')
    prev_sec.imshow(prev_container.img)
    prev_p = prev_container.traffic_light
    prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

    curr_sec.set_title('curr(' + str(curr_container.id) + ')')
    curr_sec.imshow(curr_container.img)
    curr_p = curr_container.traffic_light
    curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')
    for i in range(len(curr_p)):
        curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if curr_container.valid[i]:
            curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                          r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
    curr_sec.plot(foe[0], foe[1], 'y+')
    rot_pts = np.asarray(rot_pts)
    curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
    plt.show()


class FrameContainer(object):
    def __init__(self, img_path, id, tfls):
        self.id = id
        self.img = Image.open(img_path)
        self.traffic_light = tfls
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


class SFM_standAlone:
    def __init__(self, pkl_path):
        self.prev_frame = None
        with open(pkl_path, 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')
        self.focal = self.data['flx']
        self.pp = self.data['principle_point']
        self.curr_frame_id =0

    def change_prev_frame(self, frame_path, id, tfls):
        self.prev_frame = FrameContainer( frame_path, id, tfls)

    def run(self, tfls, curr_frame):
        if self.prev_frame is None:
            self.curr_frame_id = int(curr_frame[curr_frame.rindex('/')+1+len("dusseldorf_000049_0000"):curr_frame.rindex('/')+1+len("dusseldorf_000049_0000") + 2])
            self.prev_frame=FrameContainer(curr_frame, self.curr_frame_id, tfls)
        else:
        # prev_container = FrameContainer(prev_img_path)
            curr_container = FrameContainer(curr_frame, self.curr_frame_id, tfls)
            curr_container.traffic_light = tfls
            self.curr_frame_id+=1  #extract the frame id
            EM = np.eye(4)
            for i in range( self.prev_frame.id, self.curr_frame_id):
                EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            curr_container.EM = EM
            curr_container = calc_TFL_dist(self.prev_frame, curr_container, self.focal, self.pp)
            visualize(self.prev_frame, curr_container, self.focal, self.pp)

            self.change_prev_frame(curr_frame,self.curr_frame_id , tfls)
        print(self.curr_frame_id, ":::",tfls)
