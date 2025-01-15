#Versión Final
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans

# Cargar el modelo de Random Forest preentrenado
modelo_rf = joblib.load('C:/Users/elili/Downloads/proyectoVision/proyectoVision/modeloBaseAcercado.pkl')

# Variables globales
mueble_seleccionado = None
color_seleccionado = (0, 255, 0)  # Verde por defecto

# Cargar y mostrar la imagen
def cargar_imagen():
    global img_cv2
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    img_cv2 = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    lbl_img.config(image=img_tk)
    lbl_img.image = img_tk

# Cargar nueva imagen y extraer la paleta de colores con K-Means
def cargar_nueva_imagen():
    global img_cv2_nueva
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    img_cv2_nueva = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img_cv2_nueva, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    lbl_img_nueva.config(image=img_tk)
    lbl_img_nueva.image = img_tk

    # Extraer la paleta de colores utilizando K-Means
    paleta_colores = extraer_paleta_colores(img_cv2_nueva)
    mostrar_paleta_colores(paleta_colores)

# Extraer la paleta de colores utilizando K-Means
def extraer_paleta_colores(img_cv2):
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_rgb.shape[1] // 5, img_rgb.shape[0] // 5))  # Reducir tamaño para acelerar KMeans
    img_reshape = img_resized.reshape((-1, 3))  # Convertir la imagen a una lista de píxeles

    kmeans = KMeans(n_clusters=5)  # Número de colores a extraer
    kmeans.fit(img_reshape)

    return kmeans.cluster_centers_

# Mostrar la paleta de colores
def mostrar_paleta_colores(paleta_colores):
    global colores_paleta
    colores_paleta = paleta_colores.astype(int)

    for i, color in enumerate(colores_paleta):
        btn_color = tk.Button(ventana, bg='#%02x%02x%02x' % tuple(color), command=lambda i=i: seleccionar_color(i))
        btn_color.place(x=20 + i * 50, y=500, width=40, height=40)

# Función para seleccionar el color
# Función para seleccionar el color
def seleccionar_color(i):
    global color_seleccionado
    # Convertir de RGB (utilizado en tkinter) a BGR (utilizado en OpenCV)
    color_rgb = tuple(colores_paleta[i])
    color_seleccionado = (color_rgb[2], color_rgb[1], color_rgb[0])  # Reordenar a BGR


# Segmentación de objetos en la imagen
def segmentar_objetos(img_cv2):
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

# Clasificación de cada objeto segmentado
def clasificar_objetos(contornos):
    etiquetas = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        if w > 10 and h > 10:
            objeto = img_cv2[y:y+h, x:x+w]
            orb = cv2.ORB_create()
            _, descriptores = orb.detectAndCompute(objeto, None)

            if descriptores is not None:
                descriptores = descriptores.flatten()
                num_features_model = modelo_rf.n_features_in_
                if len(descriptores) > num_features_model:
                    descriptores = descriptores[:num_features_model]
                else:
                    descriptores = np.pad(descriptores, (0, num_features_model - len(descriptores)), 'constant')

                try:
                    prediccion = modelo_rf.predict(pd.DataFrame([descriptores]))[0]
                except ValueError:
                    prediccion = modelo_rf.predict([descriptores])[0]

                etiquetas.append((x, y, w, h, prediccion))
                print(f"Predicción: {prediccion} para el objeto en posición ({x}, {y}, {w}, {h})")
    return etiquetas

# Etiquetado de objetos en la imagen
def etiquetar_objetos(etiquetas):
    img_etiquetada = img_cv2.copy()
    for (x, y, w, h, clase) in etiquetas:
        color = (0, 255, 0) if clase == "silla" else (255, 0, 0) if clase == "mesa" else (0, 0, 255)
        cv2.rectangle(img_etiquetada, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_etiquetada, clase, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_etiquetada

# Procesamiento de la imagen para segmentación y clasificación
def procesar_imagen():
    contornos = segmentar_objetos(img_cv2)
    etiquetas = clasificar_objetos(contornos)
    img_etiquetada = etiquetar_objetos(etiquetas)

    img_rgb = cv2.cvtColor(img_etiquetada, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    lbl_img.config(image=img_tk)
    lbl_img.image = img_tk

    global mueble_seleccionado
    mueble_seleccionado = [(x, y, w, h, clase) for (x, y, w, h, clase) in etiquetas]

# Variables globales
objetos_coloreados = {}  # Diccionario para almacenar los colores de los objetos

# Evento de clic en la imagen para seleccionar un objeto
def on_click(event):
    global mueble_seleccionado, objetos_coloreados
    if mueble_seleccionado is None:
        return

    x, y = event.x, event.y
    alpha = 0.8  # Nivel de transparencia (0.0 = completamente original, 1.0 = completamente coloreado)

    for (mx, my, mw, mh, _) in mueble_seleccionado:
        if mx < x < mx + mw and my < y < my + mh:
            print(f"Objeto seleccionado: ({mx}, {my}, {mw}, {mh})")
            
            # Actualizar el color del objeto seleccionado
            objetos_coloreados[(mx, my, mw, mh)] = color_seleccionado

            # Crear una copia de la imagen base
            img_etiquetada = img_cv2.copy()

            # Repintar todos los objetos con los colores previamente seleccionados
            for (ox, oy, ow, oh), color in objetos_coloreados.items():
                fragmento = img_etiquetada[oy:oy+oh, ox:ox+ow]
                fragmento_coloreado = fragmento.copy()
                for fy in range(fragmento.shape[0]):
                    for fx in range(fragmento.shape[1]):
                        if not np.array_equal(fragmento[fy, fx], [0, 0, 0]):
                            fragmento_coloreado[fy, fx] = (
                                fragmento[fy, fx] * (1 - alpha) + np.array(color) * alpha
                            ).astype(np.uint8)
                img_etiquetada[oy:oy+oh, ox:ox+ow] = fragmento_coloreado

            # Actualizar la imagen en la interfaz
            img_rgb = cv2.cvtColor(img_etiquetada, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            lbl_img.config(image=img_tk)
            lbl_img.image = img_tk
            break




# Configuración de la interfaz
ventana = tk.Tk()
ventana.title("Aplicación de Segmentación y Etiquetado")
ventana.geometry("1000x600")

# Botones de la interfaz
btn_cargar = tk.Button(ventana, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.pack()

btn_procesar = tk.Button(ventana, text="Segmentar Imagen", command=procesar_imagen)
btn_procesar.pack()

btn_cargar_nueva = tk.Button(ventana, text="Paleta de Colores", command=cargar_nueva_imagen)
btn_cargar_nueva.pack()

# Mostrar imagen
lbl_img = tk.Label(ventana)
lbl_img.pack()

# Imagen de la paleta
lbl_img_nueva = tk.Label(ventana)
lbl_img_nueva.pack()

# Detectar clics en la imagen
lbl_img.bind("<Button-1>", on_click)

ventana.mainloop()



