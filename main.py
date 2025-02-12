import os
import time
import threading
from topCamera import topcamera
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

start_time = time.time()

# Определение функций для запуска анализа
def start_camera(rtsp_url, save_folder, FRAMES_FOLDER, api_key):
    thread = threading.Thread(target=topcamera, args=(rtsp_url, save_folder, FRAMES_FOLDER, api_key))
    thread.start()
    return thread