import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# Cargar la imagen en color
imagen = "placa.png"

placa = cv2.imread(imagen, cv2.IMREAD_COLOR)


#----------------------------------------------------------------------------


# CAPACITORES

# Preprocesamiento y aplicacion de diversas capas
placa_rgb = cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)

placa_gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilate = cv2.dilate(placa_gris, kernel, iterations=1)

# Identificación de capacitores a traves de circulos. 
circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, dp=1.4, minDist=250, param1=200, param2=130, minRadius=40, maxRadius=250)
circles = np.uint16(np.around(circles))
circles = np.array(circles)

# Crear una copia de la imagen original para dibujar los círculos
output_image_cir = cv2.cvtColor(placa_gris, cv2.COLOR_GRAY2BGR)

for circle in circles[0]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Dibujar el círculo en verde
            cv2.circle(output_image_cir, center, radius, (0, 255, 0), 5)

imshow(output_image_cir)  # Muestra la imagen con los círculos dibujados.

#----------------------------------------------------------------------------

#CHIP

dilate = cv2.dilate(placa_gris, kernel, iterations=7)
dilate = cv2.erode(placa_gris, kernel, iterations=12)
_, placa_binaria = cv2.threshold(dilate, 30, 255, cv2.THRESH_BINARY)
placa_binaria_invertida = cv2.bitwise_not(placa_binaria)
dilate = cv2.dilate(placa_binaria_invertida, kernel, iterations=4)
edges = cv2.Canny(placa_binaria_invertida, 100,0)
dilate = cv2.dilate(edges, kernel, iterations=5)

# Aplicar un umbral para convertir la imagen a binaria
_, placa_binaria = cv2.threshold(placa_gris, 90, 255, cv2.THRESH_BINARY)


# Encontrar los contornos en la imagen binaria
contornos, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
len(contornos)
# Crear una copia de la imagen original para dibujar los rectángulos
output_image_rect = placa_rgb.copy()

diagonal_chip = None
for contorno in contornos:
    # Aproximar el contorno a un polígono
    perimetro = cv2.arcLength(contorno, True)
    diagonales = cv2.approxPolyDP(contorno, 0.15 * perimetro, True)
    for i in range(len(diagonales)):
        pt1 = diagonales[i][0]
        pt2 = diagonales[(i + 1) % len(diagonales)][0]
        long_diagonal = np.linalg.norm(pt1 - pt2)
        if 600 < long_diagonal:
            diagonal_chip = (pt1, pt2)


# Asegurarse de que los puntos son numpy arrays
pt1 = np.array(diagonal_chip[0])
pt2 = np.array(diagonal_chip[1])

# Calcular las esquinas del rectángulo
top_left = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
bottom_right = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))

# Dibujar el rectángulo
cv2.rectangle(output_image_rect, top_left, bottom_right, 255, 10)  # Blanco para una imagen en escala de grises

imshow(output_image_rect)


#----------------------------------------------------------------------------


#RESISTENCIAS

# Preprocesamiento
gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (25, 25), 0)
_, binary = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)

L = 11
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(L,L))
Fd = cv2.dilate(binary, kernel, iterations=2)
edges = cv2.Canny(Fd, 10, 30)

L = 3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(L,L))
Fd = cv2.dilate(edges, kernel, iterations=1)

# Encontrar contornos
contours, _ = cv2.findContours(Fd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output_image_res = placa_rgb.copy()
resistencias = {}
puntos=[[0,0]]

# de cada contorno extraemos perimetro, diagonal (basada en una simplificacion del poligono) y area
for j, contorno in enumerate(contours):
    perimetro = cv2.arcLength(contorno, True)
    diagonal = cv2.approxPolyDP(contorno, 0.35 * perimetro, True)
    area = cv2.approxPolyDP(contorno, 0.001, True)

# medimos cada diagonal, si se ajusta al largo de las resistencias, marca su area    
    for i in range(1):
        pt1 = diagonal[i][0]
        pt2 = diagonal[(-1) % len(diagonal)][0]
        long_diagonal = np.linalg.norm(pt1 - pt2)

        if 205 < long_diagonal < 295 and all(abs(pt1 - puntos[-1]) > np.array([10, 10])):
            puntos.append(pt1)
            resistencias[j] = long_diagonal            
            cv2.drawContours(output_image_res, [area], -1, (0, 0, 255), 3)

imshow(output_image_res)  # Muestra la imagen con los contornos de las resistencias dibujados.
