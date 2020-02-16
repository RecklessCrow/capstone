import time
import mss
import mss.tools
import pytesseract
import cv2
import subprocess

game_is_running = True


# TODO Rename Screenshot file. Remove preprocess call? Access OS to get window size and location
def capture_window():
    with mss.mss() as sct:
        frame_number = 0
        while game_is_running:
            interval = 2.0 / 60.0  # running 60 fps, capture every other frame

            time.sleep(interval)  # Wait some time before taking the next screen shot

            screen_shot = f'screen_shots/frame_{frame_number}.png'

            window_pos_and_size = subprocess.check_output(["bash", "window_location_script.bash"])
            window_pos_and_size = window_pos_and_size.split(b', ')

            menu_offset = 30  # Toolbar of Mesen just looked unattractive, use this to get rid of it
            width   = int(window_pos_and_size[0])
            height  = int(window_pos_and_size[1]) - menu_offset
            x       = int(window_pos_and_size[2])
            y       = int(window_pos_and_size[3]) + menu_offset

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

            frame_number += 1


# TODO bitmap image?
# returns preprocessed image and game score
def preprocess(file_name):
    # cv2.imshow('', img)  # use to display image
    # cv2.waitKey(0)  # halts program until the '0' key is pressed so you can look at the image

    img = cv2.imread(file_name)

    # Black and white
    img = get_grayscale(img)

    # Pure Black and White
    img = thresholding(img)

    # Swap Black and White, easier to parse
    img = cv2.bitwise_not(img)

    # Smoother text0
    img = remove_noise(img)

    # These numbers are for 3x zoom on Mesen, adjust for bigger or smaller

    # Crop score out of game
    parse_score = img[200:300, 575:800]
    score = int(pytesseract.image_to_string(parse_score))

    running = True
    parse_game_over = img[350:450, 150:450]
    game_over = pytesseract.image_to_string(parse_game_over)
    if game_over.lower() == 'game over':
        running = False

    # return processed, score, running


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


capture_window()
