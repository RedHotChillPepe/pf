import math
import numpy as np

class Tracker:
    def __init__(self, max_missed=5, distance_threshold=50):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # key: object_id, value: bool (True/False)
        self.defects = {}


    def update(self, objects_rect):
        # bbox и id объектов
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            # Запускается цикл по существующим объектам, сохранённым в словаре self.center_points
            for id, pt in self.center_points.items():
                # Вычисляется евклидова дистанция между центром текущего обнаружения (cx, cy) и центром уже зарегистрированного объекта pt
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Если расстояние меньше порогового значения (35 пикселей), то считается, что обнаруженный объект совпадает с уже отслеживаемым.
                if dist < 35:
                    # Обновляем координаты центра для объекта с данным id, так как обнаружение показывает новое положение объекта.
                    self.center_points[id] = (cx, cy)
                    # В список objects_bbs_ids добавляется новый элемент – список с координатами bounding box и идентификатором объекта, который уже был ранее обнаружен.
                    objects_bbs_ids.append([x1, y1, x2, y2, id])
                    # Флаг same_object_detected устанавливается в True, что сигнализирует о том, что подобный объект найден и повторно не создаётся новый.
                    same_object_detected = True
                    # Прерывается цикл перебора уже отслеживаемых объектов, так как соответствие найдено.
                    break

            # Если после перебора существующих объектов ни один не оказался близким (флаг остался False), значит, обнаружен новый объект.
            if same_object_detected is False:
                # Новый объект добавляется в словарь center_points с ключом self.id_count и значением – его центром (cx, cy).
                self.center_points[self.id_count] = (cx, cy)
                # В список objects_bbs_ids добавляется bounding box нового объекта вместе с его новым идентификатором.
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                # В словаре self.defects для нового объекта по его идентификатору устанавливается значение False, что означает отсутствие дефекта (например, объект не помечен как дефектный).
                self.defects[self.id_count] = False
                # Увеличивается счётчик id_count, чтобы для следующего нового объекта использовать уникальный идентификатор.
                self.id_count += 1

        # Инициализируется новый пустой словарь new_center_points, который будет использован для очистки (удаления неактуальных) записей в center_points.
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            # Из списка, содержащего координаты и идентификатор, извлекается только object_id
            _, _, _, _, object_id = obj_bb_id
            # По идентификатору объекта извлекается его центр из словаря center_points.
            center = self.center_points[object_id]
            # В новый словарь new_center_points добавляется пара "идентификатор: центр", что позволяет сохранить только те объекты, для которых есть обнаружения в текущем кадре.
            new_center_points[object_id] = center

        # Старый словарь center_points заменяется обновлённым, в котором остались только актуальные объекты. Используется метод copy(), чтобы создать независимую копию.
        self.center_points = new_center_points.copy()

        # Метод возвращает список objects_bbs_ids, который содержит для каждого объекта его координаты bounding box и соответствующий идентификатор.
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