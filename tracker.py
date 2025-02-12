import math
import numpy as np

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # key: object_id, value: bool (True/False)
        self.defects = {}


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x1, y1, x2, y2, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                # Tracking
                self.defects[self.id_count] = False
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    def mark_defect(self, object_id):
        """
        Пометить яйцо как дефектное (True).
        """
        if object_id in self.defects:
            self.defects[object_id] = True

    def is_defect(self, object_id):
        """
        Проверить, было ли яйцо когда-либо дефектным.
        """
        return self.defects.get(object_id, False)

    class Track:
        def __init__(self, track_id, bbox):
            self.id = track_id
            self.bbox = bbox  # Формат: (x1, y1, x2, y2)
            self.missed = 0  # Количество последовательных кадров без обнаружения

    class EggTracker:
        def __init__(self, max_missed=5, distance_threshold=50):
            self.tracks = []  # Список активных треков
            self.next_id = 0  # Следующий id для нового трека
            self.max_missed = max_missed  # Максимальное число кадров без обнаружения, после которого трек удаляется
            self.distance_threshold = distance_threshold  # Порог для сопоставления обнаружений и треков

        def update(self, detections):
            """
            detections: список обнаруженных bounding boxes для текущего кадра [(x, y, w, h), ...]
            Возвращает список кортежей (id трека, bbox)
            """
            # Список для сопоставленных треков
            updated_tracks = []
            # Копия активных треков, чтобы отметить, с кем из них не совпали обнаружения
            unmatched_tracks = self.tracks.copy()

            # Для каждого обнаружения ищем ближайший трек
            for det in detections:
                best_track = None
                best_distance = float('inf')
                for track in unmatched_tracks:
                    dist = self._compute_distance(det, track.bbox)
                    if dist < best_distance:
                        best_distance = dist
                        best_track = track

                if best_track is not None and best_distance < self.distance_threshold:
                    # Если подходящий трек найден – обновляем его данные
                    best_track.bbox = det
                    best_track.missed = 0
                    updated_tracks.append(best_track)
                    unmatched_tracks.remove(best_track)
                else:
                    # Если не найдено подходящего трека – создаём новый
                    new_track = Track(self.next_id, det)
                    self.next_id += 1
                    updated_tracks.append(new_track)

            # Для треков, которые не получили сопоставления, увеличиваем счётчик пропусков
            for track in unmatched_tracks:
                track.missed += 1
                if track.missed <= self.max_missed:
                    updated_tracks.append(track)
            # Обновляем список активных треков
            self.tracks = [t for t in updated_tracks if t.missed <= self.max_missed]
            return [(track.id, track.bbox) for track in self.tracks]

        @staticmethod
        def _compute_distance(bbox1, bbox2):
            """
            Вычисляет евклидову дистанцию между центрами двух bounding boxes.
            bbox: (x, y, x, y)
            """
            x1, y1, x2, y2 = bbox1
            x3, y3, x4, y4 = bbox2
            center1 = (x1 + x2 / 2, y1 + y2 / 2)
            center2 = (x3 + x4 / 2, y3 + y4 / 2)
            return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Пример использования:
    if __name__ == '__main__':
        tracker = EggTracker(max_missed=5, distance_threshold=50)
        # Симуляция кадров с обнаружениями яиц (bounding boxes в формате (x1, y1, x2, y2))
        frames = [
            [(100, 100, 50, 50), (200, 200, 50, 50)],  # Кадр 1: два яйца
            [(105, 105, 50, 50)],  # Кадр 2: одно яйцо не обнаружено
            [(110, 110, 50, 50), (205, 205, 50, 50)]  # Кадр 3: яйцо появляется снова
        ]
        for i, detections in enumerate(frames):
            tracks = tracker.update(detections)
            print(f"Кадр {i + 1}:")
            for tid, bbox in tracks:
                print(f"  Трек {tid}: {bbox}")