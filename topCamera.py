import torch
import supervision as sv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import os
import time
from eggs_segment import segment_defect
import subprocess
import requests
from zoneinfo import ZoneInfo
import threading
from queue import Queue

from tracker import*
import threading
from dotenv import load_dotenv
import globals
import json

network_queue = Queue()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("supervision", sv.__version__)

load_dotenv()

model = YOLO(os.getenv('MODEL'))
# seg_model = YOLO(os.getenv('MODEL_SEGMENT'))

max_missed = int(os.getenv('MAX_MISSED'))
distance_threshold = int(os.getenv('DISTANCE_THRESHOLD'))
logs_path = os.getenv('LOGS_FOLDER')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorBGR = [x, y]
        print(colorBGR)

api_url = os.getenv('API_URL')
print(f"api_url: {api_url}")

start_time = time.time()

polygon = np.array([
            [1000, 7],
            [1131, 7],
            [1131, 1519],
            [1000, 1519]
        ])

polygon1 = np.array([
            [360, 153],
            [518, 153],
            [518, 1519],
            [360, 1519]
        ])

zone1 = np.array([
            [8, 599],
            [8, 403],
            [635, 124],
            [1268, 1],
            [1340, 189],
            [638, 359]
        ])

zone2 = np.array([
            [8, 761],
            [8, 599],
            [771, 317],
            [1737, 128],
            [1789, 369],
            [830, 540]
        ])

zone3 = np.array([
            [8, 925],
            [8, 761],
            [815, 538],
            [1791, 368],
            [1847, 635]
        ])

zone4 = np.array([
            [8, 1098],
            [8, 925],
            [1850, 635],
            [1895, 907],
            [858, 1048]
        ])

zone5 = np.array([
            [8, 1258],
            [8, 1098],
            [834, 1050],
            [1897, 902],
            [1923, 1174],
            [836, 1296]
        ])

zone6 = np.array([
            [8, 1400],
            [8, 1258],
            [852, 1294],
            [1922, 1178],
            [1945, 1422],
            [1001, 1516]
        ])

# Определяем центр зоны
zone_center = np.mean(polygon, axis=0)

def network_worker(api_url, headers):
    while True:
        data = network_queue.get()  # Ждем данные для отправки
        print(f"data.get: {data}")
        if data is None:
            break  # Специальный сигнал для завершения потока
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            print("Данные отправлены успешно. Ответ:", response.json())
        except requests.exceptions.RequestException as e:
            print("Ошибка при отправке данных:", e)
            # Можно добавить логику повторных попыток или логирование ошибки
        network_queue.task_done()

def topcamera(rtsp_url, SAVE_FOLDER, FRAMES_FOLDER, api_key):
    headers = {"Authorization": f"Api-key {api_key}"}
    network_thread = threading.Thread(target=network_worker, args=(api_url, headers), daemon=True)
    network_thread.start()
    print("Запуск камеры сверху...")
    # cap = cv2.VideoCapture(rtsp_url)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    video_info = sv.VideoInfo.from_video_path(rtsp_url)
    print("Информация о видео сверху: " + str(video_info))

    # Получение FPS (кадров в секунду) в видео
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print("FPS: " + str(fps))

    frame_time = 1 / fps

    frame_idx = 1

    zones = [
        np.array(zone1, dtype=np.int32),
        np.array(zone2, dtype=np.int32),
        np.array(zone3, dtype=np.int32),
        np.array(zone4, dtype=np.int32),
        np.array(zone5, dtype=np.int32),
        np.array(zone6, dtype=np.int32),
    ]

    tracker = Tracker(max_missed=max_missed, distance_threshold=distance_threshold)
    counting = set()
    counting_all = set()
    counting_control = set()

    counting_zone1 = set()
    counting_zone2 = set()
    counting_zone3 = set()
    counting_zone4 = set()
    counting_zone5 = set()
    counting_zone6 = set()

    # Множество ID, для которых уже сохранили фото
    already_saved_ids = set()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = 2688
    # height = 1520
    fpsv = int(cap.get(cv2.CAP_PROP_FPS))

    # Инициализация FFmpeg
    ffmpeg_cmd = [
        'ffmpeg',  # Запуск программы FFmpeg
        '-y',  # Перезаписывает файл, если он уже существует (не требуется подтверждение)
        '-f', 'rawvideo',  # Входной формат видео — сырое видео
        '-vcodec', 'rawvideo',  # Кодек для входного видео (сырой формат)
        '-pix_fmt', 'bgr24',  # Формат пикселей BGR с 24 битами на пиксель (формат OpenCV)
        '-s', f'{width}x{height}',  # Размер кадра, используемый вашим проектом (например, 1920x1080)
        '-r', str(fpsv),  # Частота кадров, соответствующая захваченному видео
        '-i', '-',  # Входное видео из стандартного ввода (поток от OpenCV)
        '-c:v', 'libx264',  # Кодек H.264 для сжатия выходного видео
        '-preset', 'ultrafast',  # Быстрый режим для уменьшения задержки, оптимально для стриминга
        '-f', 'hls',  # Формат выходного файла — HLS для трансляции в браузере
        '-hls_time', '1',  # Длина каждого сегмента HLS в секундах (для обновляемости плейлиста)
        '-hls_list_size', '5',  # Количество сегментов в плейлисте (для поддержания потока)
        '-hls_flags', 'delete_segments',  # Удаление старых сегментов для экономии места
        './stream/stream_topcamera.m3u8'  # адрес RTSP-сервера
    ]

    if ffmpeg_cmd:
        print("Инициализация ffmpeg_cmd прошла успешно")

    log_file_path = os.path.join(logs_path, "ffmpeg_debug.log")

    with open(log_file_path, "wb") as error_log:
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=error_log)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"FourCC кодек: {fourcc_str}")

    # Читаем каждый кадр
    while not globals.stop_flag:
        default_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Завершение верхней камеры: не удалось прочитать кадр.")
            break  # Завершаем, если кадры закончились

        frame = cv2.resize(frame, (width, height))

        # Пропускаем кадры с нечетным номером
        # if frame_idx % 2 != 0:
        #     frame_idx += 1
        #     continue

        counting_zone1.clear()
        counting_zone2.clear()
        counting_zone3.clear()
        counting_zone4.clear()
        counting_zone5.clear()
        counting_zone6.clear()

        zone_1 = sv.PolygonZone(polygon=zone1, frame_resolution_wh=video_info.resolution_wh)
        zone_2 = sv.PolygonZone(polygon=zone2, frame_resolution_wh=video_info.resolution_wh)
        zone_3 = sv.PolygonZone(polygon=zone3, frame_resolution_wh=video_info.resolution_wh)
        zone_4 = sv.PolygonZone(polygon=zone4, frame_resolution_wh=video_info.resolution_wh)
        zone_5 = sv.PolygonZone(polygon=zone5, frame_resolution_wh=video_info.resolution_wh)
        zone_6 = sv.PolygonZone(polygon=zone6, frame_resolution_wh=video_info.resolution_wh)
        polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
        polygon_zone_control = sv.PolygonZone(polygon=polygon1, frame_resolution_wh=video_info.resolution_wh)

        # Детекция
        results = model(frame)[0]
        detections_all = sv.Detections.from_ultralytics(results)
        # print(detections_all)
        xyxy = detections_all.xyxy

        class_ids = detections_all.class_id
        class_name = detections_all.data['class_name']
        try:
            egg_indexes = np.where(class_ids == 0)[0]
            dirty_egg_indexes = np.where(class_ids == 1)[0]
            broken_egg_indexes = np.where(class_ids == 2)[0]
        except IndexError:
            continue

        center_x = 0

        boxes_for_tracking = []
        for i in range(len(detections_all)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            boxes_for_tracking.append([x1, y1, x2, y2])

        tracked_objects = tracker.update(boxes_for_tracking)

        # Для каждого tracked_object проверяем класс и при необходимости помечаем дефект + сегментируем
        for (x1, y1, x2, y2, obj_id) in tracked_objects:

            # Ищем, какой класс у этого bbox в detections_all (проверка по совпадению координат).
            current_class_id = None
            for det_i in range(len(detections_all)):
                dx1, dy1, dx2, dy2 = map(int, xyxy[det_i])
                if abs(dx1 - x1) < 5 and abs(dy1 - y1) < 5 and abs(dx2 - x2) < 5 and abs(dy2 - y2) < 5:
                    current_class_id = class_ids[det_i]
                    break

            # Если нашли класс и он дефектный -> пометить
            if current_class_id in [1, 2, 3]:
                tracker.mark_defect(obj_id)
                counting.add(obj_id)

            # if current_class_id == 1:  # грязное яйцо
            #     # Вырезаем ROI яйца:
            #     egg_crop = frame[y1:y2, x1:x2]
            #     # cv2.imshow("egg", egg_crop)
            #
            #     # Вызываем функцию сегментации
            #     seg_result = segment_defect(egg_crop, class_id=1, seg_model=seg_model, threshold=0.2)
            #     # seg_result = None
            #
            #     if seg_result is not None:
            #         contam_percent = seg_result["contam_percent"]
            #         over_thresh = seg_result["over_threshold"]
            #         print(f"Контаминация: {contam_percent:.2f}%")
            #
            #         # Если выше нормы - передать сигнал (указать id яйца)
            #         if over_thresh:
            #             print(f"ID яйца: {obj_id}, Процент загрязнения ({contam_percent:.2f}%) выше нормы!")
            #             # здесь можно сохранить кадр, записать в лог, отправить сигнал и т.д.

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            inside_zone = False
            zone_idx = None
            for i, zone in enumerate(zones):
                # >= 0 значит точка внутри (или на границе) контура
                if cv2.pointPolygonTest(zone, (cx, cy), False) >= 0:
                    inside_zone = True
                    zone_idx = i + 1  # нумеруем зоны 1..6
                    if tracker.is_defect(obj_id) and zone_idx == 1:
                        counting_zone1.add(obj_id)
                    elif tracker.is_defect(obj_id) and zone_idx == 2:
                        counting_zone2.add(obj_id)
                    elif tracker.is_defect(obj_id) and zone_idx == 3:
                        counting_zone3.add(obj_id)
                    elif tracker.is_defect(obj_id) and zone_idx == 4:
                        counting_zone4.add(obj_id)
                    elif tracker.is_defect(obj_id) and zone_idx == 5:
                        counting_zone5.add(obj_id)
                    elif tracker.is_defect(obj_id) and zone_idx == 6:
                        counting_zone6.add(obj_id)
                    break

            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                counting_all.add(obj_id)
                # print(f"counting_all: {counting_all}")

            if cv2.pointPolygonTest(polygon1, (cx, cy), False) >= 0:
                counting_control.add(obj_id)

            # Отрисовка рамки
            if inside_zone:
                color = (95, 40, 95)  # не дефект
                if tracker.is_defect(obj_id):
                    color = (0, 0, 255)  # красный
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_x = x2 - 50
                text_y = y2 + 15
                cv2.putText(frame, f"ID {obj_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if tracker.is_defect(obj_id) and (obj_id not in already_saved_ids):
                    # Формируем имя файла
                    now_str = datetime.now().isoformat()  # формат "2025-01-19T15:32:10.123456"
                    save_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_path = os.path.join(SAVE_FOLDER, f"frame_{frame_idx}_{save_time}.jpg")
                    photo_path = os.path.join(SAVE_FOLDER, file_path)
                    cv2.imwrite(file_path, frame)

                    # Формируем запись
                    data = {
                        "id": int(obj_id),
                        "defect_type": int(current_class_id),  # 1, 2 или 3
                        "datetime": now_str,
                        "total_defects": len(counting),
                        "total_items": len(counting_all)
                        # "photo": file_path,
                        # "frame_number": frame_idx
                    }
                    print(f"data: {data}")
                    already_saved_ids.add(obj_id)

                    headers = {"Authorization": f"Api-key {api_key}"}
                    retry_count = 0
                    max_retries = 5
                    wait_time = 1

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")  # Время с миллисекундами
                    json_file_name = f"ID яйца_{int(obj_id)}.json"
                    json_file_path = os.path.join(SAVE_FOLDER, json_file_name)
                    # Сохраняем данные в JSON-файл
                    # try:
                    #     with open(json_file_path, "w", encoding="utf-8") as json_file:
                    #         json.dump(data, json_file, ensure_ascii=False, indent=4)
                    #     print(f"Данные сохранены в {json_file_path}")
                    # except Exception as save_error:
                    #     print("Ошибка при сохранении JSON файла: ", save_error)

                    if data:
                        network_queue.put(data)

        detections = detections_all[detections_all.class_id != 0]

        cv2.putText(frame, str(len(counting_all)), (35, 170), cv2.FONT_HERSHEY_COMPLEX, (4), (255, 255, 255), 5)
        cv2.putText(frame, str(len(counting_control)), (35, 300), cv2.FONT_HERSHEY_COMPLEX, (4), (0, 0, 255), 5)

        for zone_X in zones:
            cv2.polylines(frame, [zone_X], True, (0, 255, 0), 3)
        x_min1 = int(np.min(zone1[:, 0]))
        y_max1 = int(zone1[0, 1])
        cv2.putText(frame, f"Zone1: {len(counting_zone1)}", (x_min1, y_max1), cv2.FONT_HERSHEY_SIMPLEX, (1), (255, 255, 255), 2)

        x_min2 = int(np.min(zone2[:, 0]))
        y_max2 = int(zone2[0, 1])
        cv2.putText(frame, f"Zone2: {len(counting_zone2)}", (x_min2, y_max2), cv2.FONT_HERSHEY_SIMPLEX, (1),
                    (255, 255, 255), 2)

        x_min3 = int(np.min(zone3[:, 0]))
        y_max3 = int(zone3[0, 1])
        cv2.putText(frame, f"Zone3: {len(counting_zone3)}", (x_min3, y_max3), cv2.FONT_HERSHEY_SIMPLEX, (1),
                    (255, 255, 255), 2)

        x_min4 = int(np.min(zone4[:, 0]))
        y_max4 = int(zone4[0, 1])
        cv2.putText(frame, f"Zone4: {len(counting_zone4)}", (x_min4, y_max4), cv2.FONT_HERSHEY_SIMPLEX, (1),
                    (255, 255, 255), 2)

        x_min5 = int(np.min(zone5[:, 0]))
        y_max5 = int(zone5[0, 1])
        cv2.putText(frame, f"Zone5: {len(counting_zone5)}", (x_min5, y_max5), cv2.FONT_HERSHEY_SIMPLEX, (1),
                    (255, 255, 255), 2)

        x_min6 = int(np.min(zone6[:, 0]))
        y_max6 = int(zone6[0, 1])
        cv2.putText(frame, f"Zone6: {len(counting_zone6)}", (x_min6, y_max6), cv2.FONT_HERSHEY_SIMPLEX, (1),
                    (255, 255, 255), 2)

        zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.BLUE, thickness=2, text_thickness=2,
                                                 text_scale=1)
        frame = zone_annotator.annotate(scene=frame)
        zone_annotator_control = sv.PolygonZoneAnnotator(zone=polygon_zone_control, color=sv.Color.RED, thickness=2, text_thickness=2,
                                                 text_scale=1)
        frame = zone_annotator_control.annotate(scene=frame)

        # Отправка кадра в FFmpeg для трансляции
        if process and process.stdin:
            process.stdin.write(frame.tobytes())

        if counting:
            print("Список ID дефектных яиц:", counting)
            print("Количество дефектных яиц:", len(counting))
        else:
            print("Дефектных яиц не обнаружено.")

        print("Всего яиц:", len(counting_all))
        print("Всего яиц, контрольная зона:", len(counting_control))
        print(f"Номер кадра: {frame_idx}")
        frame_idx += 1

        frame_resized = cv2.resize(frame, (1920, 1080))
        # frame_resized = cv2.resize(frame, (1280, 720))
        # frame_resized = cv2.resize(frame, (2688, 1520))
        # cv2.imshow("eggs", frame_resized)
        # cv2.setMouseCallback('eggs', RGB)

        elapsed_time = time.time() - default_time
        delay = max(0, frame_time - elapsed_time)
        time.sleep(delay)
        if cv2.waitKey(1) & 0xFF == 32:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    print("Пауза верхней камеры: нажата клавиша Space.")
                    break

        if cv2.waitKey(1) & 0xFF == 27:
            print("Завершение верхней камеры: нажата клавиша Esc.")
            break

    data = {
        # "id": int(obj_id),
        # "defect_type": int(current_class_id),  # 1, 2 или 3
        "datetime": now_str,
        "total_defects": len(counting),
        "total_items": len(counting_all)
    }

    if data:
        network_queue.put(data)

    process.stdin.close()
    process.wait()
    cap.release()

    network_queue.put(None)  # Сигнал завершения
    network_thread.join()