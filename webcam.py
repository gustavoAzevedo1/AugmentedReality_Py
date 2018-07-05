import cv2
import numpy as np


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
            imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stack = np.hstack((img, cv2.cvtColor(imgCinza, cv2.COLOR_GRAY2BGR))) # compila a img em duas, uma com cor e a outra não
        cv2.imshow('webcam', stack)
        if cv2.waitKey(1) == 27:
            break  # esc para sair
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()