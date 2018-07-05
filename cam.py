# Gustavo Armborst Guedes de Azevedo
import cv2
import numpy as np
import random

# threshold do reconhecimento do objeto
MIN_MATCH_COUNT = 15

detector = cv2.xfeatures2d.SIFT_create()


# numero de frames da animação
FRAMEMAX = 1000
# movimento da grama
MOVIMENTO = 8
# numero de folhas de grama
DENSIDADE = 10
# espessura da grama
ESPEC = 2

# threshold da mascara para tirar o fundo
THRMASK = 10
# tamanho do frame da animação
TAMANHOGRM = 250

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})
FRAME = 0

gramaImg = []
lista = []
# imagem de treinamento
trainImg = cv2.imread("logo5.jpg", 0)
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
            imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # transforma a imagem para ton de cinza
            queryKP, queryDesc = detector.detectAndCompute(imgCinza, None) # função SIFT para achar pontos em comum
            matches = flann.knnMatch(queryDesc, trainDesc, k=2) # procura os pontos em comum entre a imagem treinada e da webcam

            goodMatch = []

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    goodMatch.append(m)
            if len(goodMatch) > MIN_MATCH_COUNT:   # threshold para quantidade de pontos em comum entre as imagens
                tp = []
                qp = []
                for m in goodMatch:
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp, qp = np.float32((tp, qp))
                H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
                h, w = trainImg.shape
                trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]]) # pega as bordas da imagem treinada
                queryBorder = cv2.perspectiveTransform(trainBorder, H)  # pega as bordas da imagem dentro da webcam
                # cv2.polylines(img, [np.int32(queryBorder)], True, (0, 255, 0), 5)
                img, frame = gramaWeb(img, queryBorder, frame)
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


def gramaWeb(img, borda, frame):
    if frame > FRAMEMAX:       # reseta quando chega no ultimo frame, reseta a animação
        frame = 0
    im = gramaImg[frame]
    rows, cols, ch = im.shape
    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

    pts2 = np.float32([borda[0][0], borda[0][3], borda[0][2], borda[0][1]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    rows, cols, ch = img.shape

    dst = cv2.warpPerspective(im, M, (cols, rows))  # transforma a animação para acompanhar a imagem treinada

    ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), THRMASK, 1, cv2.THRESH_BINARY_INV)  # cria uma mascara para tirar o fundo preto

    mask = cv2.erode(mask, (3, 3))
    mask = cv2.dilate(mask, (3, 3))

    for c in range(0, 3):
        img[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]   # combina as imagens e passa a mascara

    frame = frame + 1
    return img, frame


def grama():
    x = 0
    while x < TAMANHOGRM:
        lista2 = []
        lista2.append([x, (TAMANHOGRM - 150)])    # gera os potos para cada grama
        lista2.append([x, (TAMANHOGRM - 100)])
        lista2.append([x, (TAMANHOGRM - 50)])
        lista2.append([x, (TAMANHOGRM - 0)])
        lista.append(lista2)
        x = x + DENSIDADE   # quantidade de gramas


def gramado():
    x = 0
    while x < FRAMEMAX:
        aws = np.zeros((TAMANHOGRM, TAMANHOGRM, 3), np.uint8)
        for i in range(len(lista)):
            aws2 = lista[i]
            aws2[0][0] = aws2[-4][0] + random.randrange(-MOVIMENTO, MOVIMENTO)   # movimento randomico da grama, MOVIMENTO = range do movimento
            aws2[1][0] = aws2[-3][0] + random.randrange(-MOVIMENTO, MOVIMENTO)
            aws = desenha(aws, aws2)

        gramaImg.append(cv2.flip(aws, 0))   # salva a imagem em um array
        x = x + 1


def desenha(aws1, aws2):
    aws = aws1
    t = np.arange(0, 1.1, 0.1)
    lista2 = []
    for i in range(len(t)):
        ts = float(format(t[i], '.2f'))
        x = calculaBezier(ts, aws2[0][0], aws2[1][0], aws2[2][0], aws2[3][0])
        y = calculaBezier(ts, aws2[0][1], aws2[1][1], aws2[2][1], aws2[3][1])
        pts = (x, y * -1 + TAMANHOGRM)
        lista2.append(pts)
    pts = np.array(lista2, np.int32)
    cv2.polylines(aws, [pts], False, (0, 250, 0), ESPEC, 2)  # desenha as gramas baseado na função de bezier
    return aws


def main():
    grama()
    gramado()
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
