import math
import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self, max_missed, distance_threshold):
        # Словарь для хранения центров объектов: ключ — id, значение — (cx, cy)
        self.center_points = {}
        # Счетчик для присвоения уникальных id новым объектам
        self.id_count = 0

        # Словарь для хранения информации о том, был ли объект помечен как дефектный
        self.defects = {}

        # Словарь для хранения количества кадров подряд, в течение которых объект не обнаруживался
        self.missed = {}

        # Пороговое значение смещения между центрами в пикселях
        self.distance_threshold = distance_threshold
        # Максимальное число кадров, на протяжении которых объект может не обнаруживаться,
        # но при этом оставаться в трекере
        self.max_missed = max_missed

    def update(self, objects_rect):
        """
        Обновляет трек объектов по текущему списку bounding box-ов.
        Используется оптимальное сопоставление (алгоритм Венгр) между
        текущими треками и новыми детекциями. Таким образом, одно обнаружение
        сопоставляется только с одним треком, что исключает «перепрыгивание» id.

        :param objects_rect: список прямоугольников обнаруженных объектов [x1, y1, x2, y2]
        :return: список объектов с их bounding box и id: [x1, y1, x2, y2, id]
        """
        objects_bbs_ids = []
        detection_centers = []
        # Сохраним также прямоугольники, чтобы потом использовать их по индексу
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detection_centers.append((cx, cy))

        track_ids = list(self.center_points.keys())
        track_centers = [self.center_points[tid] for tid in track_ids]

        # Если есть и треки, и новые детекции, проводим оптимальное сопоставление
        if len(track_centers) > 0 and len(detection_centers) > 0:
            # Вычисляем матрицу расстояний (cost matrix)
            cost_matrix = np.zeros((len(track_centers), len(detection_centers)), dtype=np.float32)
            for i, tc in enumerate(track_centers):
                for j, dc in enumerate(detection_centers):
                    cost_matrix[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

            # Применяем алгоритм Венгр для оптимального сопоставления
            rows, cols = linear_sum_assignment(cost_matrix)

            # Множества для учёта присвоенных треков и детекций
            assigned_tracks = set()
            assigned_detections = set()

            # Пройдем по полученным парам (индекс трека, индекс детекции)
            for row, col in zip(rows, cols):
                if cost_matrix[row, col] < self.distance_threshold:
                    track_id = track_ids[row]
                    # Обновляем позицию трека – берем центр детекции
                    self.center_points[track_id] = detection_centers[col]
                    objects_bbs_ids.append(
                        [objects_rect[col][0], objects_rect[col][1], objects_rect[col][2], objects_rect[col][3],
                         track_id]
                    )
                    assigned_tracks.add(track_id)
                    assigned_detections.add(col)

            # Для треков, которым не нашлась детекция, увеличиваем счетчик пропущенных кадров
            for tid in track_ids:
                if tid not in assigned_tracks:
                    self.missed[tid] = self.missed.get(tid, 0) + 1
                else:
                    self.missed[tid] = 0

            # Детекции, которым не удалось сопоставиться с существующим треком – считаем новыми объектами
            for j, rect in enumerate(objects_rect):
                if j not in assigned_detections:
                    x1, y1, x2, y2 = rect
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    self.center_points[self.id_count] = (cx, cy)
                    self.missed[self.id_count] = 0
                    self.defects[self.id_count] = False
                    objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                    self.id_count += 1

        else:
            # Если нет существующих треков – добавляем все детекции как новые объекты
            for rect in objects_rect:
                x1, y1, x2, y2 = rect
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.center_points[self.id_count] = (cx, cy)
                self.missed[self.id_count] = 0
                self.defects[self.id_count] = False
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Удаляем треки, которые отсутствуют более max_missed кадров
        remove_ids = [tid for tid, missed in self.missed.items() if missed > self.max_missed]
        for tid in remove_ids:
            if tid in self.center_points:
                del self.center_points[tid]
            if tid in self.missed:
                del self.missed[tid]
            if tid in self.defects:
                del self.defects[tid]

        return objects_bbs_ids

    def mark_defect(self, object_id):
        """
        Помечает яйцо как дефектное (True)
        """
        if object_id in self.defects:
            self.defects[object_id] = True

    def is_defect(self, object_id):
        """
        Проверяет, был ли объект когда-либо помечен как дефектный
        """
        return self.defects.get(object_id, False)
