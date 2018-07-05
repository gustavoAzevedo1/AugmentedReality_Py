import cv2
import numpy as np
import random

TAMANHOGRM = 250

gramaImg = []
lista = []


def calculaBezier(aws, p1, p2, p3, p4):
    return ((1 - aws) ** 3 * p1
            + 3 * (1 - aws) ** 2 * aws * p2
            + 3 * (1 - aws) * aws ** 2 * p3
            + aws ** 3 * p4)


def grama():
    x = 21
    while x < 230:
        lista2 = []
        lista2.append([x, (TAMANHOGRM - 150)])
        lista2.append([x, (TAMANHOGRM - 100)])
        lista2.append([x, (TAMANHOGRM - 50)])
        lista2.append([x, (TAMANHOGRM - 0)])
        lista.append(lista2)
        x = x + 5


def gramado():
    x = 0
    while x < 500:
        aws = np.zeros((TAMANHOGRM, TAMANHOGRM, 3), np.uint8)
        for i in range(len(lista)):
            aws2 = lista[i]
            aws2[0][0] = aws2[-4][0] + random.randrange(-20, 20)
            aws2[1][0] = aws2[-3][0] + random.randrange(-20, 20)
            aws = desenha(aws, aws2)

        gramaImg.append(cv2.flip(aws, 0))
        # cv2.imshow("sdas", aws)
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
    cv2.polylines(aws, [pts], False, (0, 255, 0), 2, 2)
    return aws
