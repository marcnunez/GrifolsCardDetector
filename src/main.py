import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

UPPER_THRESHOLD_NON_ZERO = 16000
LOWER_THRESHOLD_NON_ZERO = 13000
THRESHOLD_BYNARY = 150

parser = argparse.ArgumentParser(description='Griffols Card Detector')
parser.add_argument('--debug', default=False, type=bool, help='Show histograms and images')
parser.add_argument('--threshold_non_zero', default=.25, type=float, help='Set threshold non zeros in %')
parser.add_argument('--threshold_binary', default=150, type=int, help='Set threshold between 0-256 to binaryze')
opt = parser.parse_args()


def define_ROI(image: np.ndarray)->(np.ndarray, np.ndarray):
    tl_tr = (950, 400)
    tl_bl = (450, 800)
    dimensions = (200, 100)
    roi_top_right = image[tl_tr[1]:tl_tr[1]+dimensions[1], tl_tr[0]:tl_tr[0]+dimensions[0]]
    roi_bottom_left = image[tl_bl[1]:tl_bl[1]+dimensions[1], tl_bl[0]:tl_bl[0]+dimensions[0]]
    return roi_top_right, roi_bottom_left


def RGB_to_YUV(rgb: np.ndarray)->np.ndarray:
    y = 0.299*rgb[:,:,0]+0.587*rgb[:,:,1]+0.114*rgb[:,:,2]
    y = y + np.min(y)
    y = 256*(y)/np.max(y)
    if opt.debug:
        cv2.imshow('yuv', y)
        cv2.waitKey()
        plt.hist(y.ravel(), bins=256, range=(0.0, 256.0), fc='k', ec='k')  # calculating histogram
        plt.show()
    return y


def YUV_to_bin(image: np.ndarray)->np.ndarray:
    y = (image > opt.threshold_binary) * 1.0
    if opt.debug:
        cv2.imshow('bin', y)
        cv2.waitKey()
    return y


def main():
    in_path = '../dataset/'
    for filename in os.listdir(in_path):
        frame_path = os.path.join(in_path, filename)
        image = cv2.imread(frame_path)
        roi_list = define_ROI(image)
        count_correct_roi = 0
        for roi in roi_list:
            if opt.debug:
                cv2.imshow('rgb', roi)
                cv2.waitKey()
            yuv = RGB_to_YUV(roi)
            binary = YUV_to_bin(yuv)
            non_zero = np.count_nonzero(binary)
            lower_th = binary.size-((opt.threshold_non_zero+.05)*binary.size)
            upper_th = binary.size-((opt.threshold_non_zero-.05)*binary.size)
            if lower_th < non_zero < upper_th:
                count_correct_roi += 1
        if count_correct_roi == 2:
            print("NO HAY TARGETA EN: " + filename)
        else:
            print("HAY TARGETA EN: " + filename)


if __name__ == '__main__':
    main()