import cv2
import time
import datetime
import numpy as np
import tensorflow as tf

from HandPose import doHandPoseEstimate
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork

def cashbox():
    
    ## initialize variables
    arr = [True, False, False, False]
    LOWERB = np.array([0, 0, 0])
    UPPERB = np.array([35, 35, 35])
    LOWERB_HAND = np.array([40, 60, 100])
    UPPERB_HAND = np.array([80, 100, 140])
    cap = cv2.VideoCapture('http://192.168.1.38:56506/videostream.cgi?user=admin&pwd=A2345678901')
    t = time.time()
    boxOpen = False
    handUp = False
    threshold = 70000
    thresholdHand = 1500
    startCount = False
    start = time.time()
    
    ## for handpose usage
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    ## build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    ## Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    ## pass argument
    args = {
        'image_tf': image_tf,
        'hand_side_tf': hand_side_tf,
        'evaluation': evaluation,
        'net': net,
        'hand_scoremap_tf': hand_scoremap_tf,
        'image_crop_tf': image_crop_tf,
        'scale_tf': scale_tf,
        'center_tf': center_tf,
        'keypoints_scoremap_tf': keypoints_scoremap_tf,
        'keypoint_coord3d_tf': keypoint_coord3d_tf
    }
    
    ## initialize network
    net.init(sess)

    ## main script
    while(True):    

        ret, frame = cap.read()
        frame_cpy = frame.copy()
        frame = frame[90:190, 90:270]
        mask = cv2.inRange(frame, LOWERB, UPPERB)
        maskHand = cv2.inRange(frame, LOWERB_HAND, UPPERB_HAND)

        cv2.imshow('mask', mask)
        cv2.imshow('mask_hand', maskHand)
        cv2.imshow('large frame', frame_cpy)
        cv2.imshow('small frame', frame_cpy[90:190, 90:270])

        if time.time()-t > 2 :
            t = time.time()
            print('box: ', np.count_nonzero(mask)*10)
            print('hand: ', np.count_nonzero(maskHand)*10)
            if np.count_nonzero(mask)*10 > threshold:
                if not(startCount):
                    start = time.time()
                    startCount = True
                timenow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(timenow, 'CASH BOX OPENED!' , '*ref: ', np.count_nonzero(mask)*10)
                arr[1] = True
                boxOpen = True
            if boxOpen & (np.count_nonzero(maskHand)*10 > thresholdHand):
                timenow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(timenow, 'HAND DETECTED.', '*ref: ', np.count_nonzero(maskHand))
                doHandPoseEstimate(frame, sess, args)
                arr[2] = True
                handUp = True
                startCount = False
            else:
                arr[3] = False
                handUp = False
            if boxOpen & handUp & (np.count_nonzero(mask)*10 < threshold):
                start = time.time()
                timenow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(timenow, 'CASH BOX CLOSED!' , '*ref: ', np.count_nonzero(mask)*10)
                arr[3] = True
                startCount = False
            else:
                arr[3] = False
            print(time.time() - start, startCount)
            if (time.time() - start > 10) and startCount:
                print('Warning : Cash Box opened for more than 10 secs!')
            print('Current Status : ' + str([x*1 for x in arr]))
            print('--------------------------------------------')
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
