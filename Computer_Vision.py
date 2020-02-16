import time
import mss
import mss.tools
import pytesseract
import cv2
import subprocess


# TODO Rename Screenshot file. Remove preprocess call? Access OS to get window size and location
def capture_window():
    with mss.mss() as sct:
        game_is_running = True
        frame_number = 1
        previous_score = 0
        while game_is_running:
            interval = 2.0 / 60.0  # running 60 fps, capture every other frame

            time.sleep(interval)  # Wait some time before taking the next screen shot

            screen_shot = f'screen_shots/frame_{frame_number}.png'

            window_pos_and_size = subprocess.check_output(["bash", "window_location_script.bash"])
            window_pos_and_size = window_pos_and_size.split(b', ')

            menu_offset = 30  # Toolbar of Mesen just looked unattractive, use this to get rid of it
            width = int(window_pos_and_size[0])
            height = int(window_pos_and_size[1]) - menu_offset
            x = int(window_pos_and_size[2])
            y = int(window_pos_and_size[3]) + menu_offset

            # The screen part to capture
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            output = screen_shot.format(**monitor)

            # Grab the data
            sct_img = sct.grab(monitor)

            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

            current_score, game_is_running = preprocess(screen_shot)

            # Invalid score if 0, less than the old score, or greater than double the current value
            if (current_score > (previous_score * 2) and previous_score != 0) \
                    or (current_score < previous_score) \
                    or (current_score == 0 and previous_score != 0):
                current_score = previous_score

            if current_score != 0:
                print(current_score)

            # Only keep 2 seconds of data
            if frame_number < 60:
                frame_number += 1
            else:
                frame_number = 0

            previous_score = current_score


# TODO bitmap image?
# returns preprocessed image and game score
def preprocess(file_name):
    # cv2.imshow('', img)  # use to display image
    # cv2.waitKey(0)  # halts program until the '0' key is pressed so you can look at the image

    # open screenshot
    img = cv2.imread(file_name)

    # Greyscale Image
    img = get_grayscale(img)

    # Black and White
    img = thresholding(img)

    # Swap Black and White, easier to parse
    img = cv2.bitwise_not(img)

    # Smoother text
    img = remove_noise(img)

    # These numbers are for 3x zoom on Mesen, adjust for bigger or smaller
    score_range = img[200:250, 600:]
    msg_range = img[350:450, 150:450]

    # Crop score out of game
    score = pytesseract.image_to_string(score_range, config='digits')

    # Try to cast the parse into an int
    try:
        score = int(score)
    except ValueError:
        score = 0

    running = False
    msg = pytesseract.image_to_string(msg_range)
    if msg.lower() == 'start':
        running = True
    elif msg.lower() == 'game over':
        running = False

    return score, running


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

