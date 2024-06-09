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

# Convertir la imagen de BGR a RGB
placa_rgb = cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)

# Convertir la imagen a escala de grises
placa_gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

# Definir kernel para dilatación
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Aplicar dilatación
dilate = cv2.dilate(placa_gris, kernel, iterations=1)

# Identificación de capacitores a través de círculos
circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, dp=1.4, minDist=250, param1=200, param2=130, minRadius=40, maxRadius=250)
circles = np.uint16(np.around(circles))
circles = np.array(circles)

# Crear una copia de la imagen original para dibujar los círculos
output_image_cir = cv2.cvtColor(placa_gris, cv2.COLOR_GRAY2BGR)

# Definir categorías y contadores
categories = {
    'Chico': 0,
    'Mediano-Chico': 0,
    'Mediano-Grande': 0,
    'Grande': 0
}

# Clasificación de capacitores
for circle in circles[0]:
    center = (circle[0], circle[1])
    radius = circle[2]
    
    if radius < 80:
        category = 'Chico'
        color = (255, 0, 0)  # Azul
    elif 81 <= radius < 110:
        category = 'Mediano-Chico'
        color = (0, 255, 0)  # Verde
    elif 111 <= radius < 160:
        category = 'Mediano-Grande'
        color = (0, 0, 255)  # Rojo
    else:
        category = 'Grande'
        color = (255, 255, 0)  # Amarillo
    
    # Incrementar contador de categoría
    categories[category] += 1
    
    # Dibujar el círculo y la etiqueta de clasificación
    cv2.circle(output_image_cir, center, radius, color, 5)
    cv2.putText(output_image_cir, category, (center[0] - radius, center[1] - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# Mostrar la imagen con la clasificación
imshow(output_image_cir, title="Clasificación de Capacitores")

# Mostrar el conteo de cada categoría
print("Conteo de Capacitores por Categoría:")
for category, count in categories.items():
    print(f"{category}: {count}")

