import mss
import mss.tools
import numpy as np
import pytesseract
import cv2
from PIL import Image


# TODO maybe make this into an object so that the bitmap may be accessed
# may change to pillow to loop the function efficiently
def observe(window):
    with mss.mss() as sct:

        # Grab the data
        obs = sct.grab(window)


        # Transform observation into readable image
        img = Image.frombytes("RGB", obs.size, obs.bgra, "raw", "BGRX")

        img = np.array(img)

        # Greyscale Image
        img = get_grayscale(img)

        # Swap Black and White, easier to parse
        img = cv2.bitwise_not(img)

        # Smoother text
        img = remove_noise(img)

        # Areas where information can be found
        # These numbers are for 3x zoom on Mesen, adjust for bigger or smaller
        score_range = img[200:250, 600:]
        msg_range = img[350:450, 150:450]

        # Crop score out of image
        score = pytesseract.image_to_string(score_range, config='digits')

        # Try to cast the parse into an int
        try:
            score = int(score)
        except ValueError:
            score = 0

        # crop info message out of image
        msg = pytesseract.image_to_string(msg_range)

        return obs, score, msg


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# Convert array of pixes to TF tensor
def preprocess(img):

    return img

