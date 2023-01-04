import os
import tensorflow as tf
import numpy as np
import cv2
import pytesseract
import argparse

from baseModel import DCGAN
import tensorflow_hub as hub
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def deblur(img, model_path):
    shape = (64, 64, 3)
    g = DCGAN.build_generator(shape, 16, 3)
    g.load_weights(model_path)

    pred_images = g.predict(img, verbose=0)
    pred_images = (pred_images * 127.5) + 127.5

    return pred_images

def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 final project")
    parser.add_argument("--input_path", type=str, default="test_images/", help="path to test images")
    parser.add_argument("--output", type=str, default="results/", help="path to the output images")
    parser.add_argument("--superRes", type=str, default="ESRGAN_Model", help="path to the output images")
    parser.add_argument("--deblur", type=str, default="generator.h5", help="path to the output images")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    dir = args.input_path
    path = os.listdir(dir)
    esrganModel = hub.load(args.superRes)

    imgs = []

    for p in path:
        img = cv2.imread(os.path.join(dir, p))
        imgs.append(img)

    imgs = (np.array(imgs) - 127.5) / 127.5
    deblured_image = deblur(imgs, args.deblur)
    hr_image = esrganModel(deblured_image)
    hr_image = tf.cast(hr_image, tf.uint8).numpy()

    for i in range(len(deblured_image)):
        cv2.imwrite(args.output + 'deblur/deblur_' + str(i) + ".png", deblured_image[i])
        cv2.imwrite(args.output + 'hr/hr_' + str(i) + ".png", hr_image[i])

    for i, img in enumerate(hr_image):
        out = pytesseract.image_to_string(Image.fromarray(img))
        print(str(i) + ' ::: ' + out)
