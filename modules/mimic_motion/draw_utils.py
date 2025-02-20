# Modified from mimicmotion/dwpose/util.py
import math
import numpy as np
import matplotlib
import cv2


eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, bodies):
    """
    canvas: numpy array of shape (H, W, 3)
    bodies: numpy array of shape (18, 3)
    """
    body_coords = np.array(bodies[:, :2])
    body_scores = np.array(bodies[:, 2])

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        conf = body_scores[index]
        if conf[0] < 0.3 or conf[1] < 0.3:
            continue
        Y = body_coords[index, 0]
        X = body_coords[index, 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        conf = body_scores[i]
        if conf < 0.3:
            continue
        x, y = body_coords[i]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (x, y), 4, alpha_blend_color(colors[i], conf), thickness=-1)
    return canvas

def draw_handpose(canvas, hands):
    """
    canvas: numpy array of shape (H, W, 3)
    hands: numpy array of shape (42, 3)
    """
    for hand in [hands[:21], hands[21:]]:
        hand_coords = np.array(hand[:, :2])
        hand_scores = np.array(hand[:, 2])
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        for ie, e in enumerate(edges):
            x1, y1 = hand_coords[e[0]].astype(np.int32)
            x2, y2 = hand_coords[e[1]].astype(np.int32)
            score = int(hand_scores[e[0]] * hand_scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), 
                            matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=2)
        for i, keypoint in enumerate(hand_coords):
            x, y = keypoint.astype(np.int32)
            score = int(hand_scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, face_lmks):
    """
    canvas: numpy array of shape (H, W, 3)
    face_lmks: numpy array of shape (68, 3)
    """
    lmks, scores = face_lmks[:, :2], face_lmks[:, 2]
    for lmk, score in zip(lmks, scores):
        x, y = lmk.astype(np.int32)
        conf = int(score * 255)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """
    pose: dict
    H: int
    W: int
    ref_w: int
    """
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    bodies = pose['bodies'].copy()
    hands = pose['hands'].copy()
    faces = pose['faces'].copy()
    bodies[:, :2] = bodies[:, :2] * sr
    hands[:, :2] = hands[:, :2] * sr
    faces[:, :2] = faces[:, :2] * sr
    canvas = draw_bodypose(canvas, bodies)
    canvas = draw_handpose(canvas, hands)
    canvas = draw_facepose(canvas, faces)
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB)

    
    
