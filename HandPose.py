import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from mpl_toolkits.mplot3d import Axes3D

from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

def doHandPoseEstimate(image_cv, sess, args):
    
    image_tf = args['image_tf']
    hand_side_tf = args['hand_side_tf']
    evaluation = args['evaluation']
    net = args['net']
    hand_scoremap_tf = args['hand_scoremap_tf']
    image_crop_tf = args['image_crop_tf']
    scale_tf = args['scale_tf']
    center_tf =args['center_tf']
    keypoints_scoremap_tf = args['keypoints_scoremap_tf']
    keypoint_coord3d_tf = args['keypoint_coord3d_tf']
    
    image_raw = image_cv[:,:,::-1]
    image_raw = cv2.resize(image_raw, (320, 240))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

    # visualize
    fig = plt.figure(1)
    plt.ion()
    plt.clf()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')
    ax1.imshow(image_raw)
    plot_hand(coord_hw, ax1)
    ax2.imshow(image_crop_v)
    plot_hand(coord_hw_crop, ax2)
    ax3.imshow(np.argmax(hand_scoremap_v, 2))
    plot_hand_3d(keypoint_coord3d_v, ax4)
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 1])
    ax4.set_zlim([-3, 3])
    plt.show()
    plt.pause(0.0001)
    plt.show()
