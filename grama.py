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

def show():
    while True:
        cv2.imshow("sdas", gramaImg[120])
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def main():
    grama()
    gramado()
    show()

if __name__ == '__main__':
    main()
