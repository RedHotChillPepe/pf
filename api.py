import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import main
import uvicorn
import requests
from fastapi.responses import JSONResponse
import time
from fastapi.staticfiles import StaticFiles
import globals
from typing import Optional
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from VideoStream import stream_camera
from starlette.responses import Response
import json
from datetime import datetime
import torch
from delete_ts_m3u8 import delete_files_with_extensions

# Загружаем переменные из .env файла
load_dotenv()

SAVE_FOLDER = os.getenv('SAVE_FOLDER')
print(f"SAVE_FOLDER: {SAVE_FOLDER}")
FRAMES_FOLDER = os.getenv('FRAMES_FOLDER')
print(f"FRAMES_FOLDER: {FRAMES_FOLDER}")

print(torch.__version__)  # Версия PyTorch
print(torch.cuda.is_available())  # Должно вернуть True, если CUDA доступна

app = FastAPI()

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Добавляем middleware в приложение
app.add_middleware(NoCacheMiddleware)

# Настройка для обслуживания файлов из папки `stream`
app.mount("/stream", StaticFiles(directory="stream"), name="stream")

api_url = os.getenv('API_URL')
print(f"api_url: {api_url}")

# Модель данных для POST-запроса
class CameraRequest(BaseModel):
    api_key: str
    rtsp_url: str

# Глобальная переменная для отслеживания потоков
threads = []

@app.post("/start_video", status_code=201)
def start_video(request: CameraRequest):
    folder_path = './stream'
    extensions_to_delete = ['.ts', '.m3u8']
    delete_files_with_extensions(folder_path, extensions_to_delete)
    print("Полученный АПИ ключ для входа:", request.api_key)
    print("Полученный URL основной камеры:", request.rtsp_url)

    if not request.api_key:
        raise HTTPException(status_code=401, detail="API-ключ не может быть пустым")

    # Запуск потоков для каждой камеры
    thread1 = threading.Thread(target=stream_camera, args=(request.rtsp_url, request.api_key))

    # Запуск потоков
    thread1.start()

    return {"detail": "Видеопоток запущен без обработки"}

@app.post("/start_analysis", status_code=201)
def start_analysis(request: CameraRequest):
    folder_path = './stream'
    extensions_to_delete = ['.ts', '.m3u8']
    delete_files_with_extensions(folder_path, extensions_to_delete)
    print("Полученный АПИ ключ для входа:", request.api_key)
    print("Полученный URL основной камеры:", request.rtsp_url)
    global threads

    if not request.api_key:
        raise HTTPException(status_code=401, detail="API-ключ не может быть пустым")

    if len(threads) > 0:
        stop_analysis()  # Останавливаем текущий анализ, если он уже запущен
        raise HTTPException(status_code=409, detail="Анализ уже был запущен")

    # Сбрасываем флаг остановки
    globals.stop_flag = False

    # Запуск анализа в отдельных потоках
    thread1 = threading.Thread(target=start_camera_with_flag, args=(request.rtsp_url, request.api_key))

    thread1.start()

    # Добавляем потоки в список для управления
    threads.extend([thread1])

    return {"detail": "Нейросеть запущена для анализа видеопотока"}

@app.post("/stop_analysis", status_code=200)
def stop_analysis():
    global threads

    if len(threads) == 0:
        print("Запрос на остановку анализа, но потоки отсутствуют.")
        raise HTTPException(status_code=404, detail="Анализ не был запущен")

    print(f"Попытка остановить анализ. Активных потоков: {len(threads)}")

    # Устанавливаем флаг для остановки потоков
    globals.stop_flag = True

    # Ждем завершения всех потоков
    for thread in threads:
        if thread.is_alive():
            thread.join()

    # Проверяем, если после join потоки все еще активны, выводим предупреждение
    for thread in threads:
        if thread.is_alive():
            raise HTTPException(status_code=500,
                                detail="Не удалось корректно остановить анализ. Потоки все еще активны.")

    # Очищаем список потоков
    threads.clear()
    # return {"status": "Анализ остановлен"}

    # Ожидаем завершения всех потоков анализа
    time.sleep(2)  # Задержка для завершения текущих операций

    return {"detail": "Анализ остановлен, начата трансляция без анализа"}

# Маршрут для получения ссылок на HLS-потоки
@app.get("/stream/camera", response_class=JSONResponse)
def aside_camera_stream():
    return {"url": "http://<your-domain>/stream/stream_camera.m3u8"}

# Функции с проверкой на флаг остановки
def start_camera_with_flag(rtsp_url, api_key):
    headers = {"Authorization": f"Api-key {api_key}"}
    retry_count = 0
    max_retries = 5
    wait_time = 1

    while not globals.stop_flag:
        main.topcamera(rtsp_url, SAVE_FOLDER, FRAMES_FOLDER, api_key)

# Запуск FastAPI
if __name__ == "__main__":
    try:
        uvicorn.run(app, host=os.getenv('HOST'), port=int(os.getenv('PORT')))
    except KeyboardInterrupt:
        globals.stop_flag = True
        print("Сервер остановлен")