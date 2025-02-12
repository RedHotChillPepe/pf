import supervision as sv
import cv2
import os
import time
import subprocess
from dotenv import load_dotenv
import globals

# Загружаем переменные из .env файла
load_dotenv()

api_url = os.getenv('API_URL')
print(f"api_url: {api_url}")

start_time = time.time()

def stream_camera(rtsp_url, api_key):
    print(f"URL основной камеры: {rtsp_url}, API Key: {api_key}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    video_info = sv.VideoInfo.from_video_path(rtsp_url)
    print("Информация о видео: " + str(video_info))

    # Получение FPS (кадров в секунду) в видео
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print("FPS: " + str(fps))
    # Индекс для сохраняемых кадров
    frame_idx = 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    with open("ffmpeg_debug.log", "wb") as error_log:
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=error_log)

    # Читаем каждый кадр
    while not globals.stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Завершение основной камеры: не удалось прочитать кадр.")
            break  # Завершаем, если кадры закончились

        frame = cv2.resize(frame, (width, height))

        # Отправка кадра в FFmpeg для трансляции
        if process and process.stdin:
            process.stdin.write(frame.tobytes())

        # print(f"Номер кадра: {frame_idx}")
        frame_idx += 1

        frame_resized = cv2.resize(frame, (1274, 720))
        # cv2.imshow("asidecamera", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Завершение камеры: нажата клавиша Esc.")
            break

    process.stdin.close()
    process.wait()
    cap.release()