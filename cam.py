import cv2
import numpy as np

MIN_MATCH_COUNT = 30

detector = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread("logo.jpg", 0)
trainImg = cv2.flip(trainImg, 1)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
            # img = cv2.resize(img, (1080, 720))
            imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            queryKP, queryDesc = detector.detectAndCompute(imgCinza, None)
            matches = flann.knnMatch(queryDesc, trainDesc, k=2)

            goodMatch = []

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    goodMatch.append(m)
            if len(goodMatch) > MIN_MATCH_COUNT:
                tp = []
                qp = []
                for m in goodMatch:
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp, qp = np.float32((tp, qp))
                H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
                h, w = trainImg.shape
                trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                cv2.polylines(img, [np.int32(queryBorder)], True, (0, 255, 0), 5)
            cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc para sair
    cam.release()
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()