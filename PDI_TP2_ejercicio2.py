import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lista de nombres de archivos de imágenes a procesar
imagenes = [
    "img01.png", "img02.png", "img03.png", "img04.png",
    "img05.png", "img06.png", "img07.png", "img08.png", 
    "img09.png", "img10.png", "img11.png", "img12.png"
]

autos = []
crop_patentes = []

# Cargar las imágenes y almacenarlas en 'autos'
for i in imagenes:
    auto = cv2.imread(i, cv2.IMREAD_COLOR)
    if auto is not None:
        autos.append(auto)

# Procesar cada imagen en la lista para detectar placas
for auto in autos:
    # Obtener dimensiones de la imagen
    altura, anchura, _ = auto.shape

    # Definir áreas de recorte para concentrarse en zonas típicas donde se localizan las placas
    corte_superior = int(altura * 0.25)
    corte_inferior = int(altura * 0.80)
    corte_izquierdo = int(anchura * 0.30)
    corte_derecho = int(anchura * 0.75)

    # Recortar la imagen para enfocarse potencialmente en la placa
    imagen_recortada = auto[corte_superior:corte_inferior, corte_izquierdo:corte_derecho]
    imagen_gris = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2GRAY)

    # Obtener la mediana de intensidad de la imagen en escala de grises
    mediana_original = np.median(imagen_gris)
    mediana = mediana_original.copy()
    bandera = True  # Bandera para controlar el bucle while
    
    while bandera:
        # Ajustar el umbral para la binarización basado en la mediana
        umbral_inferior = mediana + 81 # Dentro del While la mediana se va ajustanto para que ecuentre todas las placas
        _, imagen_umbral = cv2.threshold(imagen_gris, umbral_inferior, 255, cv2.THRESH_BINARY)

        # Detectar componentes conectados en la imagen binarizada.
        _, etiquetas, estadisticas,_ = cv2.connectedComponentsWithStats(imagen_umbral)
        #etiquetas: Es una matriz del mismo tamaño que la imagen de entrada, 
        #donde cada elemento tiene un valor que representa la etiqueta del componente conectado al que pertenece
        #estadisticas: Es la matriz Stats 

        # Definir rangos de área para identificar componentes de tamaño plausible de placa
        umbral_area = 17
        umbral_area2 = 300
        mascara_componentes = np.zeros_like(imagen_umbral)
        for i in range(1, etiquetas.max() + 1):
            area = estadisticas[i, cv2.CC_STAT_AREA]
            if area > umbral_area and area < umbral_area2:
                mascara_componentes[etiquetas == i] = 255

        # Filtrar componentes por la relación de aspecto (en base al alto y ancho de los objetos)
        num_labels, labels_1, stats, _ = cv2.connectedComponentsWithStats(mascara_componentes)
        labels_aspect_ratio_filtered = labels_1.copy()
        for i in range(num_labels):
            if stats[i, 3] / stats[i, 2] < 1.5 or stats[i, 3] / stats[i, 2] > 3:
                labels_aspect_ratio_filtered[labels_aspect_ratio_filtered == i] = 0


        # Revisar y ordenar los componentes filtrados
        num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(labels_aspect_ratio_filtered.astype(np.uint8))
        indices_ordenados_por_x = np.argsort(stats_final[:, cv2.CC_STAT_LEFT])
        stats_ordenadas = stats_final[indices_ordenados_por_x]

        # Analizar los gaps entre componentes conectados para identificar patrones de placas
        resultados = []
        for i in range(2, num_labels_final):
            fila_actual = stats_ordenadas[i]
            fila_anterior = stats_ordenadas[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])
            resultados.append(suma)

        # Recortar con dimensiones en torno de la placa
        alto_recorte, ancho_recorte = 60, 100

        # Buscar un gap específico entre componentes que sugiera la presencia de una placa
        # El gap es la distancia entre los grupos de letras y números de una patente
        for i in range(1, len(resultados)):
            if 8 <= resultados[i] <= 14:
                bandera = False  # Se encontró un gap adecuado, detener el bucle

                # Determinar las coordenadas para el recorte final basado en componentes conectados
                # Lo siguiente hace que a partir del gap se aplique el recorte de dimensiones en 60, 100
                x1, y1, w1, h1, _ = stats_ordenadas[i+1] 
                x2, y2, w2, h2, _ = stats_ordenadas[i + 2]
                punto_medio_x = x1 + w1 + ((x2 - (x1 + w1)) // 2) + corte_izquierdo
                punto_medio_y = (y1 + h1 // 2 + y2 + h2 // 2) // 2 + corte_superior

                inicio_x = max(0, punto_medio_x - ancho_recorte // 2)
                fin_x = inicio_x + ancho_recorte
                inicio_y = max(0, punto_medio_y - alto_recorte // 2)
                fin_y = inicio_y + alto_recorte

                inicio_x = min(inicio_x, auto.shape[1] - ancho_recorte)
                inicio_y = min(inicio_y, auto.shape[0] - alto_recorte)

                # Recortar la imagen de la placa y almacenar en la lista 'crop_patentes'
                imagen_recortada_final = auto[inicio_y:fin_y, inicio_x:fin_x]
                crop_patentes.append(imagen_recortada_final)


                ## Visualización de la imagen en escala de grises
                #plt.figure()
                #plt.imshow(imagen_gris, cmap='gray')  # Asumiendo que es una imagen en escala de grises
                #plt.title('Primer crop')
                #plt.axis('off')  # Opcional: elimina los ejes
                #plt.show()
#
                ## Visualización de la imagen umbralizada
                #plt.figure()
                #plt.imshow(imagen_umbral, cmap='gray')  # Asumiendo que también es una imagen en escala de grises
                #plt.title('Imagen umbralizada')
                #plt.axis('off')
                #plt.show()
#
                ## Visualización de la primera máscara
                #plt.figure()
                #plt.imshow(mascara_componentes, cmap='gray')
                #plt.title('Primera máscara')
                #plt.axis('off')
                #plt.show()
#
                ## Visualización de la máscara filtrada por aspecto
                #plt.figure()
                #plt.imshow(labels_aspect_ratio_filtered, cmap='gray')
                #plt.title('Máscara filtrada por aspecto')
                #plt.axis('off')
                #plt.show()
#
                ## Visualización del segundo crop, convertido de BGR a RGB para correcta visualización de colores
                #plt.figure()
                #plt.imshow(cv2.cvtColor(imagen_recortada_final, cv2.COLOR_BGR2RGB))
                #plt.title('Segundo crop')
                #plt.axis('off')
                #plt.show()
#
                #print()
                #print(stats_ordenadas)
                #print()
                #print(resultados)

                break  
    
        mediana -= 2  # Ajustar la mediana para cambiar el umbral si no se encontró el gap deseado


# Iterar sobre cada imagen de placa en la lista crop_patentes

patentes_procesadas = []

for patente in crop_patentes:

    # Crear una copia de la imagen para procesar
    placa = patente.copy()
    
    # Suavizar la imagen usando un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(placa, (5, 5), 0)
    
    # Incrementar la nitidez de la imagen original mezclándola con la versión desenfocada
    sharpened = cv2.addWeighted(placa, 1.5, blurred, -0.5, 0)
    
    # Convertir la imagen a escala de grises para simplificar el procesamiento
    img_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    
    # Aplicar CLAHE para mejorar el contraste de la imagen en áreas locales (cualización de histograma)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img_gray)

    # Calcular la mediana de los valores de intensidad para establecer un umbral inicial
    mediana = np.median(img_eq)
    umbral_inferior = mediana 
    num_labels_final = None

    # Repetir el umbralizado aumentando el umbral hasta encontrar 7 componentes conectados
    while num_labels_final != 7:

        # Umbralizar la imagen para crear una binarización
        _, imagen_umbral = cv2.threshold(img_eq, umbral_inferior, 255, cv2.THRESH_BINARY)
        
        # Obtener componentes conectados y sus estadísticas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_umbral)
        
        # Definir los umbrales de área para los componentes
        umbral_area = 8
        umbral_area2 = 75
        
        # Crear una máscara inicial donde solo se incluyen los componentes de áreas deseadas
        mascara_componentes = np.zeros_like(imagen_umbral)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if umbral_area < area < umbral_area2:
                mascara_componentes[labels == i] = 255

        # Filtrar componentes por la relación de aspecto (en base al alto y ancho de los objetos)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_componentes)
        labels_aspect_ratio_filtered = labels.copy()
        for i in range(1, num_labels):
            aspect_ratio = stats[i, cv2.CC_STAT_HEIGHT] / stats[i, cv2.CC_STAT_WIDTH]
            if not (1.5 < aspect_ratio < 3):
                labels_aspect_ratio_filtered[labels == i] = 0

        # Contar los componentes conectados después de filtrar por relación de aspecto
        num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(labels_aspect_ratio_filtered.astype(np.uint8))
        indices_ordenados_por_x = np.argsort(stats_final[:, cv2.CC_STAT_LEFT])
        stats_ordenadas = stats_final[indices_ordenados_por_x]
        
        # Incrementar el umbral si no se encontraron exactamente 7 componentes
        umbral_inferior += 1

    # Dibujar bounding boxes rojos alrededor de los componentes filtrados en la imagen original
    for i in range(1, num_labels_final):
        x = stats_ordenadas[i, cv2.CC_STAT_LEFT]
        y = stats_ordenadas[i, cv2.CC_STAT_TOP]
        w = stats_ordenadas[i, cv2.CC_STAT_WIDTH]
        h = stats_ordenadas[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(placa, (x, y), (x + w, y + h), (0, 0, 255), 1) 
    
    # Visualización de la imagen umbralizada
    #plt.figure()
    #plt.imshow(imagen_umbral, cmap='gray')  
    #plt.title('Imagen umbralizada')
    #plt.axis('off')
    #plt.show()
    
    ## Visualización de la primera máscara
    #plt.figure()
    #plt.imshow(mascara_componentes, cmap='gray')
    #plt.title('Primera máscara')
    #plt.axis('off')
    #plt.show()

    ## Visualización de la máscara filtrada por aspecto
    #plt.figure()
    #plt.imshow(labels_aspect_ratio_filtered, cmap='gray')
    #plt.title('Máscara filtrada por aspecto')
    #plt.axis('off')
    #plt.show()
    #plt.figure()

    #plt.imshow(cv2.cvtColor(placa, cv2.COLOR_BGR2RGB))
    #plt.title('Placa con Bounding Boxes')
    #plt.axis('off')
    #plt.show()

    ## Calcular la anchura y altura promedio de todos los bounding boxes
    anchuras = [stats_ordenadas[i, cv2.CC_STAT_WIDTH] for i in range(1, num_labels_final)]
    anchura_promedio = np.mean(anchuras)
    desplazamiento_horizontal = anchura_promedio // 2
    
    alturas = [stats_ordenadas[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels_final)]
    altura_promedio = np.mean(alturas)
    desplazamiento = altura_promedio // 2

    # Coordenadas del primer y último bounding box
    x1, y1, w1, h1 = stats_ordenadas[1, cv2.CC_STAT_LEFT], stats_ordenadas[1, cv2.CC_STAT_TOP], stats_ordenadas[1, cv2.CC_STAT_WIDTH], stats_ordenadas[1, cv2.CC_STAT_HEIGHT]
    x2, y2, w2, h2 = stats_ordenadas[-1, cv2.CC_STAT_LEFT], stats_ordenadas[-1, cv2.CC_STAT_TOP], stats_ordenadas[-1, cv2.CC_STAT_WIDTH], stats_ordenadas[-1, cv2.CC_STAT_HEIGHT]

    # Puntos medios de los bounding boxes del primer y último componente
    punto_medio_1 = (x1 + w1 // 2, y1 + h1 // 2)
    punto_medio_2 = (x2 + w2 // 2, y2 + h2 // 2)

    # Coordenadas de los puntos extendidos y ajuste para bounding box verde
    puntos = [
        (punto_medio_1[0] - int(desplazamiento_horizontal*2.3), punto_medio_1[1] - int(desplazamiento*2.3)),  # Superior izquierdo
        (punto_medio_2[0] + int(desplazamiento_horizontal*2.3), punto_medio_2[1] - int(desplazamiento*2.3)),  # Superior derecho
        (punto_medio_2[0] + int(desplazamiento_horizontal*2.3), punto_medio_2[1] + int(desplazamiento*2.3)),  # Inferior derecho
        (punto_medio_1[0] - int(desplazamiento_horizontal*2.3), punto_medio_1[1] + int(desplazamiento*2.3))   # Inferior izquierdo
    ]

    # Dibujar el bounding box en la imagen
    pts = np.array(puntos, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(placa, [pts], True, (0, 255, 0), 2)  # Dibujar un polígono cerrado

    # Dibujar los puntos azules desplazados 
    for punto in puntos:
        cv2.circle(placa, punto, 3, (0, 0, 255), -1)  # Azul

    patentes_procesadas.append(placa)  # Añadir la placa procesada a la lista

# Configurar el tamaño de la figura y el arreglo de subplots (3 columnas, 4 filas)
fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # Ajusta el tamaño según necesites

# Colocar un título general para la figura
fig.suptitle('Patentes encontradas', fontsize=16)

# Aplanar el arreglo de ejes para facilitar el acceso en un bucle
axs = axs.flatten()

# Mostrar cada imagen en su respectivo subplot
for ax, img in zip(axs[:len(patentes_procesadas)], patentes_procesadas):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

# Ocultar los ejes de los subplots que no tienen imagen
for ax in axs[len(patentes_procesadas):]:
    ax.axis('off')

plt.show()
