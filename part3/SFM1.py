import math

import numpy as np

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (len(norm_prev_pts) == 0):
        print('no prev points')
    elif (len(norm_prev_pts) == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    normalized = []
    for tfl in pts:
        x = (tfl[0]-pp[0])/focal
        y = (tfl[1]-pp[1])/focal
        normalized.append([x,y])
    return normalized
    # transform pixels into normalized pixels using the focal length and principle point
    
def unnormalize(pts, focal, pp):
    unnormalized = []
    for tfl in pts:
        x = tfl[0] * focal + pp[0]
        y = tfl[1] * focal + pp[1]
        unnormalized.append([x, y])
    return unnormalized
    # transform normalized pixels into pixels using the focal length and principle point

def decompose(EM):
    R=EM[0:-1, 0:-1]
    t=EM[0:-1,-1]
    tZ=t[2]
    foe=(t[0]/t[2],t[1]/t[2])
    return R,foe, tZ
    # extract R, foe and tZ from the Ego Motion

def rotate(pts, R):
    rotated=[]
    for tfl in pts:
        tfl=np.asarray([tfl[0],tfl[1], 1])
        vector=np.dot(R, tfl)
        rotated.append([vector[0]/vector[2], vector[1]/vector[2]])
    return rotated
    # rotate the points - pts using R

def find_corresponding_points(p, norm_pts_rot, foe):
    m = (foe[1]-p[1])/(foe[0]-p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0]-p[0])
    min_index, distance = 0, abs((m * norm_pts_rot[0][0] + n - norm_pts_rot[0][1]) / math.sqrt(math.pow(m, 2) + 1))
    for i in range(1,len(norm_pts_rot)):
        curr_dist = abs(m * norm_pts_rot[i][0] + n - norm_pts_rot[i][1]) / math.sqrt(math.pow(m, 2) + 1)
        if curr_dist < distance:
            min_index, distance = i, curr_dist
    return min_index, (norm_pts_rot[min_index])

    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index

def calc_dist(p_curr, p_rot, foe, tZ):
    Zx = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    Zy = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])
    x_gap = abs(p_curr[0]-p_rot[0])
    y_gap = abs(p_curr[1]-p_rot[1])
    return (Zx * x_gap + Zy * y_gap ) / (x_gap + y_gap)
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z