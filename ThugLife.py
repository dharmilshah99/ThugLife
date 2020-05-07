import cv2
import dlib
import numpy as np
from imutils import face_utils
from math import atan2, degrees, sin, cos
import argparse


def resize_sticker(sticker, points, error=1.7):
    """
    Resize sticker based on the distance between two co-ordinates.

    Parameters
    ----------
    sticker (array): Array (2D or greater) representing the sticker's image.
    points (array): Pair of co-ordinates in an array. Eg. [left_x, left_y, right_x and right_y]
    error (float): Error term to account for whitespace within image.
    
    Returns
    -------
    sticker_resized (array): Array representing the sticker resized.
    """
    # Shape of image to resize.
    img_h, img_w = sticker.shape[0:2]
    # Obtain new shape of image.
    points_copy = points.copy()
    size_w = int(np.linalg.norm(points_copy[2:4] - points_copy[0:2]) * error)
    size_h = int(img_h * (size_w / img_w))
    # Resize.
    sticker_resized = cv2.resize(sticker, (size_w, size_h))
    return sticker_resized


def rotate_sticker(sticker, points):
    """
    Rotates sticker based on the angle between the two co-ordinates.

    Parameters
    ----------
    sticker (array): Array (2D or greater) representing the sticker's image.
    points (array): Pair of co-ordinates in an array. Eg. [left_x, left_y, right_x and right_y]

    Returns
    -------
    sticker_rotated (array): Array representing the sticker rotated.
    """
    # Obtain angle between points. Account for inverted numpy y-axis.
    y_diff = points[3] - points[1]
    x_diff = points[2] - points[0]
    angle = atan2(y_diff * -1, x_diff)
    # Find new image size after rotation.
    h, w = sticker.shape[0:2]
    sin_theta = abs(sin(angle))
    cos_theta = abs(cos(angle))
    new_h = int(h * cos_theta + w * sin_theta)
    new_w = int(h * sin_theta + w * cos_theta)
    # Translate image to new center and resize.
    m_translation = np.float32([[1, 0, (new_w // 2 - w // 2)], [0, 1, (new_h // 2 - h // 2)]])
    sticker_translated = cv2.warpAffine(sticker, m_translation, (new_w, new_h))
    # Rotate image about new center.
    m_rotation = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), degrees(angle), 1)
    sticker_rotated = cv2.warpAffine(sticker_translated, m_rotation, (new_w, new_h))
    return sticker_rotated


def super_impose(sticker, image, points):
    """
    Places sticker onto image given points.

    Parameters
    ----------
    sticker (array): Array (2D or greater) representing the sticker's image.
    points (array): Pair of co-ordinates in an array. Eg. [left_x, left_y, right_x and right_y]

    Returns
    -------
    image (array): Array representing the image with the sticker superimposed.
    """
    # Obtain center between points.
    cent_w, cent_h = (points[0] + points[2]) / 2, (points[1] + points[3]) / 2
    # With the sticker centered here, obtain its bounds on the image.
    sticker_h, sticker_w = sticker.shape[0:2]
    w_lower, w_upper = int(cent_w - sticker_w / 2), int(cent_w + sticker_w / 2)
    h_lower, h_upper = int(cent_h - sticker_h / 2), int(cent_h + sticker_h / 2)
    region = image[h_lower:h_upper, w_lower:w_upper]
    # Using sticker's Alpha channel, mask visible areas on region.
    r, g, b, a = cv2.split(sticker)
    background = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(a))
    # For masked areas, inherit colors from the sticker.
    foreground = cv2.bitwise_and(cv2.merge((b, g, r)), cv2.merge((b, g, r)), mask=a)
    # Add.
    image[h_lower:h_upper, w_lower:w_upper] = cv2.add(background, foreground)
    return image


def parser():
    """
    Command line interface to obtain path of image to play with.

    Returns
    -------
    image_path (str): Path to image.
    """
    # Arguments.
    parser = argparse.ArgumentParser(description='Thug Life your Image.')
    parser.add_argument('-path', type=str, default=None, help='Path to the faces you want to Thug Life.')
    # Parse.
    args = parser.parse_args()
    image_path = args.path
    return image_path


if __name__ == '__main__':
    # Parse Arguments.
    image_path = parser()

    # Load face Bounding Box detector and Facial Keypoint detector.
    bb_detector = dlib.get_frontal_face_detector()
    kp_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Load Images.
    image = cv2.imread(image_path)
    sticker = cv2.imread('transparent.png', -1)

    # Predict BBs.
    bounding_boxes = bb_detector(image, 1)

    # For every BB, predict facial key points and place sticker.
    for box in bounding_boxes:
        # Obtain facial key points of interest.
        points = kp_detector(image, box)
        points = face_utils.shape_to_np(points)
        points_oi = np.concatenate((points[36], points[45]))
        points_oi = points_oi.reshape(4, )
        # Sticker transformations.
        sticker_rotated = rotate_sticker(sticker, points_oi)
        sticker_resized = resize_sticker(sticker_rotated, points_oi)
        # Super impose.
        image = super_impose(sticker_resized, image, points_oi)

    # Display.
    cv2.imshow('Thug Life!', image)
    cv2.waitKey(0)
