import cv2
import numpy as np
import random

nome = 'B-Spline  ---  BackSpace -> Limpa a tela      Space -> Nova Spline    ESC -> Sair'
TM = 800  # tamanho da tela
img = np.zeros((TM, int(TM * 2.25), 3), np.uint8)
cv2.namedWindow(nome)
spline = 1  # B-Spline -> 0     Bezier -> 1
lista = []


def ponto(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img, 'P{}'.format(len(lista)), (x - 10, y - 7), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        p = [x, (TM - y)]  # salva o ponto clicado e passa o y para cardinal
        lista.append(list.copy(p))
        print("P{} >> x = {} e y = {}".format(len(lista) - 1, lista[-1][0], lista[-1][1]))
        if len(lista) >= 4:
            t = np.arange(0, 1.1, 0.1)
            lista2 = []
            for i in range(len(t)):
                aws = float(format(t[i], '.2f'))
                if spline == 1:
                    x = calculaBezier(aws, lista[-4][0], lista[-3][0], lista[-2][0], lista[-1][0])
                    y = calculaBezier(aws, lista[-4][1], lista[-3][1], lista[-2][1], lista[-1][1])
                if spline == 0:
                    x = calculaBspline(aws, lista[-4][0], lista[-3][0], lista[-2][0], lista[-1][0])
                    y = calculaBspline(aws, lista[-4][1], lista[-3][1], lista[-2][1], lista[-1][1])

                pts = (x, y * -1 + TM)
                lista2.append(pts)
            pts = np.array(lista2, np.int32)
            r, g, b = random.randrange(40, 255), random.randrange(40, 255), random.randrange(40, 255)
            cv2.polylines(img, [pts], False, (b, g, r), 2, 8)


cv2.setMouseCallback(nome, ponto)


def main():
    while True:
        cv2.imshow(nome, img)
        teste = cv2.waitKey(1)
        if teste == 27:
            break
        if teste == 8:
            limpaTela()
        if teste == 32:
            zeraLista()
    cv2.destroyAllWindows()


def calculaBspline(aws, p1, p2, p3, p4):
    return (((1 - aws) ** 3) / 6 * p1
            + (3 * (aws ** 3) - 6 * (aws ** 2) + 4) / 6 * p2
            + (-3 * (aws ** 3) + 3 * (aws ** 2) + 3 * aws + 1) / 6 * p3
            + (aws ** 3) / 6 * p4)


def calculaBezier(aws, p1, p2, p3, p4):
    return ((1 - aws) ** 3 * p1
            + 3 * (1 - aws) ** 2 * aws * p2
            + 3 * (1 - aws) * aws ** 2 * p3
            + aws ** 3 * p4)


def limpaTela():
    cv2.rectangle(img, (0, 0), (TM * 100, TM * 100), (0, 0, 0), -1)
    lista.clear()

def troca(x):
    spline = x

def zeraLista():
    lista.clear()


if __name__ == "__main__":
    main()
