import mss
import mss.tools
import pytesseract
import cv2


# TODO Rename Screenshot file. Remove preprocess call?
def capture_window(running):
    with mss.mss() as sct:
        while running:
            screen_shot = 'test.png'

            monitor_number = 1
            mon = sct.monitors[monitor_number]

            # Used zoom x3 on mesen
            width = 255 * 3
            height = 240 * 3

            # The screen part to capture
            monitor = {
                "top": mon["top"] + 85,
                "left": mon["left"],
                "width": width,
                "height": height,
                "mon": monitor_number,
            }
            output = screen_shot.format(**monitor)

            # Grab the data
            sct_img = sct.grab(monitor)

            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

            preprocess(screen_shot)


# TODO bitmap image?
# returns preprocessed image and game score
def preprocess(file_name):
    img = cv2.imread(file_name)

    # Black and white
    img = get_grayscale(img)

    cv2.imshow("", img)
    cv2.waitKey(0)

    # Pure Black and White
    img = thresholding(img)

    # Swap Black and White, easier to parse
    img = cv2.bitwise_not(img)

    # Smoother text
    img = remove_noise(img)

    # Crop score out of game
    parse_score = img[200:300, 575:800]
    score = int(pytesseract.image_to_string(parse_score))

    # return processed, score


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
