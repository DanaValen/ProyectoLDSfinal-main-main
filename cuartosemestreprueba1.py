import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = (int(centroid[0]), int(centroid[1]))
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializamos la captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definimos los puntos clave de las manos
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Definimos colores con valores BGR
PEACH = (255, 153, 102)
PURPLE = (153, 102, 204)

# Inicializamos un diccionario para almacenar el conteo de dedos para cada mano
fingers_counters = {0: "_", 1: "_"}

# Inicializamos una matriz de espesores de líneas para los rectángulos de las manos
thickness = [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]

# Iniciamos el contexto de Mediapipe para el seguimiento de manos
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Volteamos el fotograma horizontalmente para que coincida con el espejo
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            # Si se detectan manos
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []

                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])

                # Calculamos el ángulo del pulgar
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                
                thumb_finger = angle > 150

                # Calculamos el centroide de la palma
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)
                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counters[idx] = str(np.count_nonzero(fingers == True))
                for i, finger in enumerate(fingers):
                    if finger:
                        thickness[idx][i] = -1

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Mostramos el número de dedos en la pantalla
        cv2.putText(frame, f"Fingers: {fingers_counters.get(0, '_')}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Fingers: {fingers_counters.get(1, '_')}", (370, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualización de los rectángulos de las manos
        for idx, (hand_color, hand_thickness) in enumerate(zip([PEACH, PURPLE], [thickness[0], thickness[1]])):
            # Pulgar
            cv2.rectangle(frame, (160 + idx * 200, 10), (210 + idx * 200, 60), hand_color, hand_thickness[0])
            cv2.putText(frame, "Pulgar", (160 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Índice
            cv2.rectangle(frame, (220 + idx * 200, 10), (270 + idx * 200, 60), hand_color, hand_thickness[1])
            cv2.putText(frame, "Indice", (220 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Medio
            cv2.rectangle(frame, (280 + idx * 200, 10), (330 + idx * 200, 60), hand_color, hand_thickness[2])
            cv2.putText(frame, "Medio", (280 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Anular
            cv2.rectangle(frame, (340 + idx * 200, 10), (390 + idx * 200, 60), hand_color, hand_thickness[3])
            cv2.putText(frame, "Anular", (340 + idx * 200, 80), 1, 1, (255, 255, 255), 2)
            # Meñique
            cv2.rectangle(frame, (400 + idx * 200, 10), (450 + idx * 200, 60), hand_color, hand_thickness[4])
            cv2.putText(frame, "Meñique", (400 + idx * 200, 80), 1, 1, (255, 255, 255), 2)

        # Mostramos el fotograma
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberamos los recursos y cerramos la ventana
cap.release()
cv2.destroyAllWindows()
