import cv2
import numpy as np
import random

MIN_MATCH_COUNT = 30

detector = cv2.xfeatures2d.SIFT_create()

FRAME = 0

TAMANHOGRM = 250

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

gramaImg = []
lista = []
trainImg = cv2.imread("logo.jpg", 0)
trainImg = cv2.flip(trainImg, 1)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)


def show_webcam(mirror=False):
    frame = FRAME
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
                # cv2.polylines(img, [np.int32(queryBorder)], True, (0, 255, 0), 5)
                img, frame = gramaWeb(img, trainBorder, frame)
            cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc para sair
    cam.release()
    cv2.destroyAllWindows()


def calculaBezier(aws, p1, p2, p3, p4):
    return ((1 - aws) ** 3 * p1
            + 3 * (1 - aws) ** 2 * aws * p2
            + 3 * (1 - aws) * aws ** 2 * p3
            + aws ** 3 * p4)


def grama():
    x = 5
    while x < 230:
        lista2 = []
        lista2.append([x, (TAMANHOGRM - 200)])
        lista2.append([x, (TAMANHOGRM - 100)])
        lista2.append([x, (TAMANHOGRM - 50)])
        lista2.append([x, (TAMANHOGRM - 0)])
        lista.append(lista2)
        x = x + 10


def gramado():
    x = 0
    while x < 500:
        for i in range(len(lista)):
            aws = np.zeros((TAMANHOGRM, TAMANHOGRM), np.uint8)
            aws2 = lista[i]
            aws2[-4][0] = aws2[-4][0] + random.randrange(-20, 20)
            aws2[-3][0] = aws2[-3][0] + random.randrange(-20, 20)
            gramaImg.append(desenha(aws, aws2))
        x = x + 1


def gramaWeb(img, borda, frame):
    if frame > 500:
        frame = 0

    rows, cols, ch = TAMANHOGRM, TAMANHOGRM, 3
    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    # pts2 is used for defining the perspective transform
    # pts2 = np.float32([borda[0][0], borda[0][3], borda[0][2], borda[0][1]])
    # compute the transform matrix
    M = cv2.getPerspectiveTransform(pts1, borda)
    # M = borda
    rows, cols, ch = img.shape
    # make the perspective change in a image of the size of the camera input
    dst = cv2.warpPerspective(gramaImg[frame], M, (cols, rows))
    # A mask is created for adding the two images
    # maskThreshold is a variable because that allows to substract the black background from different images
    ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), 10, 1, cv2.THRESH_BINARY_INV)
    # Erode and dilate are used to delete the noise
    mask = cv2.erode(mask, (3, 3))
    mask = cv2.dilate(mask, (3, 3))
    # The two images are added using the mask
    for c in range(0, 3):
        img[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]

    # novoArray = np.float32([borda[0][0], borda[0][3], borda[0][2], borda[0][1]])
    # resultado = cv2.getPerspectiveTransform(np.float32(gramaImg[frame]), novoArray)
    # resultadoImg = cv2.warpPerspective(gramaImg[frame], resultado, (640, 480))  # resolucao da minha web cam
    # resultadoFinalImg = cv2.addWeighted(img, 1, resultadoImg, 2, 0)

    frame = frame + 1
    return img, frame


def desenha(aws, lista):
    t = np.arange(0, 1.1, 0.1)
    lista2 = []
    for i in range(len(t)):
        ts = float(format(t[i], '.2f'))
        x = calculaBezier(ts, lista[-4][0], lista[-3][0], lista[-2][0], lista[-1][0])
        y = calculaBezier(ts, lista[-4][1], lista[-3][1], lista[-2][1], lista[-1][1])
        pts = (x, y * -1 + TAMANHOGRM)
        lista2.append(pts)
    pts = np.array(lista2, np.int32)
    cv2.polylines(aws, [pts], False, (0, 255, 0), 2, 8)
    return aws


def main():
    grama()
    gramado()
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
