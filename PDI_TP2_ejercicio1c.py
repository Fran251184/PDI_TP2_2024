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
placa_rgb = cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)

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

# Mostrar la imagen con las resistencias detectadas
imshow(output_image_res, new_fig=True, title="Resistencias Detectadas", color_img=True)

# Mostrar subplots de cada resistencia segmentada
fig, axs = plt.subplots(1, len(resistencias), figsize=(15, 5))
for i, (idx, long_diagonal) in enumerate(resistencias.items()):
    x, y, w, h = cv2.boundingRect(contours[idx])
    resistencia_img = placa_rgb[y:y+h, x:x+w]
    axs[i].imshow(resistencia_img)
    axs[i].set_title(f'R {i+1}')
    axs[i].axis('off')

plt.show()

# Mostrar el resultado calculado por consola
print(f"Cantidad de resistencias eléctricas presentes en la placa: {len(resistencias)}")

