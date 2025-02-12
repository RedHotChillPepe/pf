import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# seg_model = YOLO(os.getenv('MODEL_SEGMENT'))

def segment_defect(egg_crop, class_id, seg_model, threshold=0.2):
    """
    Функция для сегментации грязного яйца (class_id=1) и подсчёта процента загрязнения.

    Параметры:
    ----------
    egg_crop : np.ndarray
        ROI (изображение) яйца, вырезанное из исходного кадра.
    class_id : int
        Класс яйца (0 - чистое, 1 - грязное, 2 - битое и т.д.).
    seg_model : объект модели
        Ваша модель сегментации (предполагаем, что она умеет принимать
        egg_crop и выдавать маску загрязнения).
    threshold : float
        Порог, выше которого считаем, что яйцо слишком загрязнено (в долях или процентах).

    Возвращает:
    -----------
    result : dict or None
        Словарь с полями:
         - "egg_area": площадь всего яйца (в пикселях)
         - "contam_area": площадь загрязнения (в пикселях)
         - "contam_percent": процент загрязнения (0..100)
         - "over_threshold": bool (True/False)
        Если класс != 1 (т.е. не грязное) -- можно вернуть None
        или аналогичную структуру с нулевыми полями.
    """

    # Если яйцо не грязное (class_id=1), можем сразу сказать None
    # или вернуть пустой результат
    if class_id != 1:
        return None

    # 1) Запуск модели сегментации.
    #    Предположим, что seg_model(egg_crop) вернёт нам маску 0..1
    #    того же размера, где 1 = загрязнённая область.
    results = seg_model.predict(egg_crop)
    # print(f"results: {results}")
    # Допустим, mask — это np.ndarray float32, значения в диапазоне [0..1].
    # Нужно порогнуть её.
    masks = results[0].masks
    print(f"masks: {masks}")
    if masks != None:
        mask_i = results[0].masks.data[i].cpu().numpy()
        binary_mask = (mask_i > 0.5).astype(np.uint8)

        # 2) Считаем площади (в пикселях)
        contam_area = np.sum(binary_mask)
        egg_area = egg_crop.shape[0] * egg_crop.shape[1]

        # 3) Рассчитываем процент загрязнения
        contam_percent = (contam_area / egg_area) * 100.0

        # 4) Смотрим, превышает ли допустимый порог
        over_threshold = (contam_percent > (threshold * 100.0))

        result = {
            "egg_area": egg_area,
            "contam_area": contam_area,
            "contam_percent": contam_percent,
            "over_threshold": over_threshold
        }
        return result

    else:
        print("Masks is None")