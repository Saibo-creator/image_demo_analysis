"""1. Getting Started with Pre-trained Model from InsightFace
=======================================================


In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`insightface-model-zoo`
and analyze faces from images.

Step by Step
------------------

Let's first try out a pre-trained insightface model with a few lines of python code.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``insightface`` if you haven't done so yet.
"""

import insightface
import urllib
import urllib.request
import cv2
import os
import numpy as np
from PIL import Image
import glob
import logging
from matplotlib.image import imread

logger = logging.getLogger()

age_groups = {
    'childhood': range(0, 8),
    'puberty': range(8, 13),
    'adolescence': range(13, 18),
    'adulthood': range(18, 30),
    'middle_age': range(30, 50),
    'seniority': range(50, 120),
}
age_ranges = [1, 8, 13, 18, 30, 50, 120]


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def load_image_to_array(filepath):
    data = cv2.imread(filepath)
    return data


def build_image_age_dict(path, delimiter="|"):
    paths = glob.glob(path + '/**/*.jpg', recursive=True)
    age_dict = {}
    for filepath in paths:
        # find age label for WIKIAge
        filename, ext = os.path.splitext(os.path.basename(filepath))
        try:
            lbl_age = float(filename.split(delimiter)[-1])
        except Exception as e:
            logger.warn(e)
            continue
        for lbl, irange in age_groups.items():
            if min(irange) <= lbl_age <= max(irange):
                break
        age_dict[filepath] = lbl_age

    return age_dict


def detect_age(img: np.ndarray, model, ctx_id=-1):
    """
    # Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs
    """
    model.prepare(ctx_id=ctx_id, nms=0.4)
    faces = model.get(img)
    assert len(faces) == 1
    for idx, face in enumerate(faces):
        print("\tage:%d" % (face.age))
        return face.age


def show_img_label(img, caption: str):
    img_text = cv2.putText(img,
                           caption,
                           (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           2,
                           (0, 255, 0),
                           3,
                           cv2.LINE_AA)
    cv2.imshow("Image with Label", img_text)
    cv2.waitKey(0)


if __name__ == '__main__':

    ################################################################
    #
    # Then, we download and show the example image:
    #
    # url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
    # img = url_to_image(url)
    img_root_path = "puberty clean"

    age_dict = build_image_age_dict(img_root_path, delimiter="_")

    model = insightface.app.FaceAnalysis()

    paths = glob.glob(img_root_path + '/**/*.jpg', recursive=True)
    for img_path in paths:
        img = load_image_to_array(img_path)
        age = detect_age(img, model=model)
        caption = f"True age={age_dict[img_path]}, Predict age={age}"
        show_img_label(img, caption)
