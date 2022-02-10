import pyrealsense2 as rs
import numpy as np
import cv2

# Abrimos
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Iniciamos el stream
pipeline.start(config)


def DibujarContornos(imagen, contornos, color, Palabra):
    for c in contornos:
        M = cv2.moments(c)
        if (M["m00"] == 0): M["m00"] = 1
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        cv2.drawContours(imagen, [c], 0, color, 2)
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(imagen, approx, 0, color, 2)
        cv2.putText(imagen, str(Palabra), (x, y), 1, 2, (0, 0, 0), 2)


try:
    while True:
        frames = pipeline.wait_for_frames()  # Obtiene frame
        color = frames.get_color_frame()
        if not color:
            continue
        color_image = np.asanyarray(color.get_data())
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)  # Convertimos BGR  a HSV para deteccion de colores

        # Comenzamos a definir los limites del color rojo
        Red_Min = (165, 90, 80)  # HSV
        Red_Max = (180, 255, 255)  # HSV

        # Comenzamos a definir los limites del color Azul
        BLue_Min = (90, 90, 80)  # HSV
        Blue_Max = (135, 255, 255)  # HSV
        # Hacemos la discriminacion de valores que no pertenecen al rojo
        mascaraR = cv2.inRange(hsv_image, Red_Min,
                               Red_Max)  # verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Red_Output = cv2.bitwise_and(color_image, color_image,
                                     mask=mascaraR)  # la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        # SECCION DEL AZUL
        mascaraB = cv2.inRange(hsv_image, BLue_Min,
                               Blue_Max)  # verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Blue_Output = cv2.bitwise_and(color_image, color_image,
                                      mask=mascaraB)  # la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        # # Contornos Rojos
        # ContornoRojo = cv2.findContours(mascaraR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        # DibujarContornos(color_image, ContornoRojo, (255, 255, 255), "Equipo")
        # Contornos Azules
        ContornoAzul = cv2.findContours(mascaraB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        DibujarContornos(color_image, ContornoAzul, (255, 255, 255), "Enemigo")

        cv2.imshow('RGB', color_image)
        cv2.imshow('Rojo', Red_Output)
        cv2.imshow('Azul', Blue_Output)

        cv2.waitKey(1)  # muestra la imagen en N milisegundos

finally:
    pipeline.stop()
