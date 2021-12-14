# !pip install opencv-python==3.4.2.17
# !pip install opencv-contrib-python==3.4.2.17
import os
import argparse
import logging
import cv2
import numpy as numpy
def combine_images(img0, img1, h_matrix):
    points0 = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=numpy.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points0, points2), axis=0)
    [x_min, y_min] = numpy.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img


def is_cv2():
    major, minor, increment = cv2.__version__.split(".")
    return major == "2"


def is_cv3():
    major, minor, increment = cv2.__version__.split(".")
    return major == "3"


def display(title, img, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def save_image(path, result):
    name, ext = os.path.splitext(path)
    img_path = '{0}.png'.format(name)
    cv2.imwrite(img_path, result)


def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1
    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)
    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)
    src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))
    return src_pts, dst_pts, len(positive)





knn=2
lowe=0.7
min_correspondence=10
sift = cv2.xfeatures2d.SIFT_create()
result = None
result_gry = None
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 6}, {'checks': 50})
image_paths = "/img"
image_index = -1



filename = sorted(os.listdir("img"))
no_of_images=len(filename)

for i in range(no_of_images):
    if "jpg" in  filename[i]:
        print(filename[i])
        image_colour = cv2.imread("img/"+filename[i])
        image_colour = image_colour.astype('uint8')
        image_gray = cv2.cvtColor(image_colour, cv2.COLOR_RGB2GRAY)
        image_index += 1
        if image_index == 0:
            result = image_colour
            result_gry = image_gray
            continue
        features0 = sift.detectAndCompute(result_gry, None)
        features1 = sift.detectAndCompute(image_gray, None)
        matches_src, matches_dst, n_matches = compute_matches(features0, features1, flann,knn=knn)
        H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
        result =  combine_images(image_colour, result, H)
        result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        # display("Output",result)
        save_image(str(i)+".jpg", result)
