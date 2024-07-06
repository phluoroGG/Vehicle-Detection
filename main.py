import math
import sys

import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMenuBar, QAction, \
    QSlider, QActionGroup, QHBoxLayout, QScrollArea, QLineEdit
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtGui import QImage
import cv2
from torchvision.ops import nms, box_convert
from torchvision.transforms.functional import pil_to_tensor
from munkres import Munkres

import Model

mnkrs = Munkres()
classes = []


class DetectedObject:
    counter = 1

    def __init__(self, class_, center):
        self.name = "object {}".format(self.counter)
        DetectedObject.counter += 1
        self.trajectory = [center]
        self.class_ = [0 for _ in range(len(classes))]
        self.class_[classes.index(class_)] += 1

    def get_name(self):
        return self.name

    def get_class(self):
        return classes[self.class_.index(max(self.class_))]

    def get_trajectory(self):
        return self.trajectory

    def get_last_position(self):
        return self.trajectory[-1]

    def add_point(self, class_, center):
        self.trajectory.append(center)
        self.class_[classes.index(class_)] += 1

    @staticmethod
    def clear_counters():
        DetectedObject.counter = 1


class VideoPlayer(QWidget):

    def __init__(self):
        super().__init__()

        self.model_name = None
        self.setWindowTitle("Обнаружение техники")
        self.video_path = ""
        self.video_capture = None
        self.is_playing = False
        self.current_frame = None
        self.start_frame = 0
        self.model = None
        self.prev_objects = []
        self.objects = []
        self.labels = []
        self.pos = None
        self.pixmap = None

        self.height = None
        self.angleSlope = None
        self.angleVert = None
        self.angleHoris = None
        self.lat = None
        self.lon = None
        self.angleDirection = None

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.mousePressEvent = self.get_pos

        self.play_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderReleased.connect(self.play_video)
        self.slider.sliderMoved.connect(self.set_frame)
        self.slider.sliderPressed.connect(self.stop_video)

        self.play_button.clicked.connect(self.play_video)
        self.stop_button.clicked.connect(self.stop_video)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollContent = QWidget()

        self.scrollLayout = QVBoxLayout()
        scrollContent.setLayout(self.scrollLayout)

        detections_layout = QVBoxLayout()
        detections_layout.addWidget(scroll)

        scroll.setWidget(scrollContent)

        self.scrollLayout.setAlignment(Qt.AlignTop)
        scroll.setMinimumSize(200, 20)
        scroll.setMaximumSize(200, 3000)

        label1 = QLabel()
        label1.setText("Введите высоту камеры (в метрах):")
        self.e1 = QLineEdit()
        self.e1.setMinimumSize(200, 20)
        self.e1.setMaximumSize(200, 3000)
        label2 = QLabel()
        label2.setText("Введите угол наклона камеры:")
        self.e2 = QLineEdit()
        self.e2.setMinimumSize(200, 20)
        self.e2.setMaximumSize(200, 3000)
        label3 = QLabel()
        label3.setText("Введите вертикальный угол \n обзора камеры:")
        self.e3 = QLineEdit()
        self.e3.setMinimumSize(200, 20)
        self.e3.setMaximumSize(200, 3000)

        label4 = QLabel()
        label4.setText("Введите горизонтальный угол \n обзора камеры:")
        self.e4 = QLineEdit()
        self.e4.setMinimumSize(200, 20)
        self.e4.setMaximumSize(200, 3000)
        label5 = QLabel()
        label5.setText("Введите широту (Формат: ГГ.ГГГГ):")
        self.e5 = QLineEdit()
        self.e5.setMinimumSize(200, 20)
        self.e5.setMaximumSize(200, 3000)
        label6 = QLabel()
        label6.setText("Введите долготу (Формат: ГГ.ГГГГ):")
        self.e6 = QLineEdit()
        self.e6.setMinimumSize(200, 20)
        self.e6.setMaximumSize(200, 3000)
        label7 = QLabel()
        label7.setText("Введите угол направления камеры \n относительно востока:")
        self.e7 = QLineEdit()
        self.e7.setMinimumSize(200, 20)
        self.e7.setMaximumSize(200, 3000)

        self.confirm_button = QPushButton("Подтвердить")
        self.confirm_button.clicked.connect(self.accept_coords)
        self.confirm_button.setMinimumSize(200, 20)
        self.confirm_button.setMaximumSize(200, 3000)

        settings_layout = QVBoxLayout()
        settings_layout.setAlignment(Qt.AlignTop)
        settings_layout.addWidget(label1)
        settings_layout.addWidget(self.e1)
        settings_layout.addWidget(label2)
        settings_layout.addWidget(self.e2)
        settings_layout.addWidget(label3)
        settings_layout.addWidget(self.e3)

        settings_layout.addWidget(label4)
        settings_layout.addWidget(self.e4)
        settings_layout.addWidget(label5)
        settings_layout.addWidget(self.e5)
        settings_layout.addWidget(label6)
        settings_layout.addWidget(self.e6)
        settings_layout.addWidget(label7)
        settings_layout.addWidget(self.e7)

        settings_layout.addWidget(self.confirm_button)

        data_layout = QHBoxLayout()
        data_layout.addLayout(detections_layout)
        data_layout.addWidget(self.video_label)
        data_layout.addLayout(settings_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(data_layout)
        main_layout.addWidget(self.play_button)
        main_layout.addWidget(self.stop_button)
        main_layout.addWidget(self.slider)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.timeout.connect(self.update_slider_position)

        self.create_menu()

    def accept_coords(self):
        try:
            self.height = float(self.e1.text())
            self.angleSlope = float(self.e2.text())
            self.angleVert = float(self.e3.text())
            self.angleHoris = float(self.e4.text())
        except:
            pass
        try:
            self.lat = float(self.e5.text())
            self.lon = float(self.e6.text())
            self.angleDirection = float(self.e7.text())
        except:
            pass

    def get_pos(self, event):
        if self.start_frame != 0:
            pos = event.pos()
            width_label = self.video_label.frameGeometry().width()
            height_label = self.video_label.frameGeometry().height()
            height, width, channel = self.current_frame.shape
            coeff = max(width / width_label, height / height_label)
            x_offset = (width_label - width / coeff) / 2  # 0 if coeff = width / width_label
            y_offset = (height_label - height / coeff) / 2  # 0 if coeff = height / height_label
            self.pos = ((pos.x() - x_offset) / width * coeff,
                        (pos.y() - y_offset) / height * coeff)
            self.update()

    def paintEvent(self, paint_event):
        if self.start_frame != 0:
            self.display_frame(self.current_frame)
            painter = QPainter(self.video_label.pixmap())
            pen = QPen()
            pen.setWidth(8)
            pen.setColor(QColor('red'))
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing, True)
            if self.pos is not None:
                width_label = self.video_label.frameGeometry().width()
                height_label = self.video_label.frameGeometry().height()
                height, width, channel = self.current_frame.shape
                coeff = max(width / width_label, height / height_label)
                pos = QPoint(int(self.pos[0] * width / coeff),
                             int(self.pos[1] * height / coeff))
                painter.drawPoint(pos)

    def create_menu(self):
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("Файл")
        open_action = QAction("Открыть", self)
        open_action.triggered.connect(self.select_video)
        file_menu.addAction(open_action)
        model_menu = menu_bar.addMenu("Модель")
        select_no_model = QAction("Без модели", self)
        select_no_model.triggered.connect(self.clear_model)
        model_menu.addAction(select_no_model)
        select_detr1 = QAction("DETR UAV", self)
        select_detr1.triggered.connect(self.create_model_detr1)
        model_menu.addAction(select_detr1)
        select_detr2 = QAction("DETR Vehicle 1", self)
        select_detr2.triggered.connect(self.create_model_detr2)
        model_menu.addAction(select_detr2)
        select_detr3 = QAction("DETR Vehicle 2", self)
        select_detr3.triggered.connect(self.create_model_detr3)
        model_menu.addAction(select_detr3)
        select_detr4 = QAction("DETR Vehicle 3", self)
        select_detr4.triggered.connect(self.create_model_detr4)
        model_menu.addAction(select_detr4)
        export_objects = menu_bar.addMenu("Экспорт")
        export_txt = QAction("Сохранить .txt координаты в метрах", self)
        export_txt.triggered.connect(self.export_txt)
        export_objects.addAction(export_txt)
        save_txt_lat_lon = QAction("Сохранить .txt координаты широты и долготы", self)
        save_txt_lat_lon.triggered.connect(self.save_txt_lat_lon)
        export_objects.addAction(save_txt_lat_lon)
        save_video = menu_bar.addMenu("Сохранить видео")
        save_action = QAction("Только видео", self)
        save_action.triggered.connect(self.save_video)
        save_video.addAction(save_action)
        save_with_txt_action = QAction("Вместе с .txt координатами в метрах", self)
        save_with_txt_action.triggered.connect(self.save_video_txt)
        save_video.addAction(save_with_txt_action)
        save_with_txt_action = QAction("Вместе с .txt координатами широты и долготы", self)
        save_with_txt_action.triggered.connect(self.save_video_txt_lat_lon)
        save_video.addAction(save_with_txt_action)

        alignment_group = QActionGroup(self)
        alignment_group.addAction(select_no_model)
        alignment_group.addAction(select_detr1)
        alignment_group.addAction(select_detr2)
        alignment_group.addAction(select_detr3)
        select_no_model.setChecked(True)
        self.layout().setMenuBar(menu_bar)

    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Выбрать видео", "", "Видеофайлы (*.avi *.mp4 *.flv)")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.is_playing = False
            self.current_frame = None
            self.pos = None
            self.start_frame = 0
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.clear_info()

    def clear_info(self):
        self.objects.clear()
        self.prev_objects.clear()
        DetectedObject.clear_counters()
        self.update_labels()

    def clear_model(self):
        self.model = None
        self.model_name = None

    def create_model_detr1(self):
        self.model = Model.ModelCompilation("detr1")
        self.model_name = 'detr1'
        global classes
        classes = ['airplane', 'drone', 'helicopter', 'uav']
        self.clear_info()

    def create_model_detr2(self):
        self.model = Model.ModelCompilation("detr2")
        self.model_name = 'detr2'
        global classes
        classes = ['vehicle']
        self.clear_info()

    def create_model_detr3(self):
        self.model = Model.ModelCompilation("detr3")
        self.model_name = 'detr3'
        global classes
        classes = ['small-vehicle', 'large-vehicle']
        self.clear_info()

    def create_model_detr4(self):
        self.model = Model.ModelCompilation("detr4")
        self.model_name = 'detr4'
        global classes
        classes = ['vehicle']
        self.clear_info()

    def stop_video(self):
        self.is_playing = False
        self.timer.stop()
        if self.video_capture:
            self.start_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            self.video_capture.release()
            self.video_capture = None

    def play_video(self):
        if self.video_path and not self.is_playing:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.is_playing = True
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(fps))

    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
        else:
            self.stop_video()
            return

        processed_frame = self.process_frame(self.current_frame)
        self.current_frame = processed_frame
        self.update_labels()
        self.display_frame(processed_frame)

    def process_frame(self, frame, iou_threshold=0.1):
        if self.model_name == 'detr1':
            size = (640, 512)
            prob_threshold = 0.2
        elif self.model_name == 'detr2':
            size = (1024, 1024)
            prob_threshold = 0.2
        elif self.model_name == 'detr3':
            size = (840, 712)
            prob_threshold = 0.2
        elif self.model_name == 'detr4':
            size = (1024, 512)
            prob_threshold = 0.2
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = frame_pil.size
        resized_frame = frame_pil.resize(size, resample=4)
        img_tensor = pil_to_tensor(resized_frame) / 255
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_tensor)
        bbox = predictions['pred_boxes'][0]
        scores = predictions['pred_logits'][0].softmax(1)
        scores = scores[:, 0:len(classes)]
        out, idxs = torch.max(scores, dim=1)
        bbox = bbox[out > prob_threshold]
        idxs = idxs[out > prob_threshold]
        out = out[out > prob_threshold]
        centers = bbox[:, 0:2]
        bbox = box_convert(torch.as_tensor(bbox), 'cxcywh', 'xyxy')
        for box in bbox:
            box[0] = int(box[0] * frame.shape[1])
            box[1] = int(box[1] * frame.shape[0])
            box[2] = int(box[2] * frame.shape[1])
            box[3] = int(box[3] * frame.shape[0])
        bbox_idx = nms(bbox, out, iou_threshold).detach().cpu().numpy()
        bbox = bbox[bbox_idx]
        idxs = idxs[bbox_idx]
        centers = centers[bbox_idx]
        frame_pil = np.array(frame_pil)
        distance_matrix = []
        for i, val1 in enumerate(self.prev_objects):
            for j, val2 in enumerate(centers):
                if len(distance_matrix) < i + 1:
                    distance_matrix.append([])
                if len(distance_matrix[i]) < j + 1:
                    distance_matrix[i].append([])
                distance_matrix[i][j] = np.linalg.norm(
                    np.array(self.objects[val1].get_last_position()) - np.array(val2.cpu()))
        current_objects = []
        if len(self.prev_objects) == 0:
            for i in range(len(centers)):
                current_objects.append(len(self.objects))
                self.objects.append(DetectedObject(classes[idxs[i]], centers[i].tolist()))
        elif distance_matrix:
            best_idxs = mnkrs.compute(distance_matrix)
            best_idxs = [list(i) for i in best_idxs]
            curr_idxs = [i[1] for i in best_idxs]
            for i in range(len(centers)):
                try:
                    index = curr_idxs.index(i)
                    if distance_matrix[best_idxs[index][0]][best_idxs[index][1]] > 0.1:
                        current_objects.append(len(self.objects))
                        self.objects.append(DetectedObject(classes[idxs[i]], centers[i].tolist()))
                    else:
                        current_objects.append(self.prev_objects[best_idxs[index][0]])
                        self.objects[self.prev_objects[best_idxs[index][0]]].add_point(classes[idxs[i]],
                                                                                       centers[i].tolist())
                except ValueError:
                    current_objects.append(len(self.objects))
                    self.objects.append(DetectedObject(classes[idxs[i]], centers[i].tolist()))
        self.prev_objects = current_objects
        if width < 500 or height < 500:
            font_scale = 0.7
            thickness = 1
        elif width < 1500 or height < 1500:
            font_scale = 0.7
            thickness = 1
        else:
            font_scale = 1
            thickness = 2
        for i, box in enumerate(bbox):
            cv2.rectangle(frame_pil, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), thickness)
            cv2.putText(frame_pil, self.objects[current_objects[i]].get_name(), (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 0), thickness)
        return frame_pil

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def set_frame(self):
        if self.video_capture:
            frame_number = self.slider.value()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = None

    def update_slider_position(self):
        if self.video_capture:
            frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(frame_number)

    def update_labels(self):
        for labe in self.labels:
            labe.deleteLater()
        self.labels.clear()
        for i in self.prev_objects:
            object_ = self.objects[i]
            lab = QLabel()
            if self.height is not None and self.angleSlope is not None and self.angleVert is not None and self.angleHoris is not None:
                lab.setText("{}\n{}\n{}\n{}\n\n".format(object_.get_name(), object_.get_class(),
                                                        *self.search_dx_dy(object_.get_last_position()[0],
                                                                           object_.get_last_position()[1])))
            else:
                lab.setText("{}\n{}\n{}\n{}\n\n".format(object_.get_name(), object_.get_class(),
                                                        object_.get_last_position()[0], object_.get_last_position()[1]))
            lab.setStyleSheet("font-size: 16px")
            lab.setMinimumSize(200, 80)
            lab.setAlignment(Qt.AlignTop)
            self.labels.append(lab)
            self.scrollLayout.addWidget(lab)

    def _export_txt(self, path):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Экспортировать объекты", "", "Текстовые файлы (*.txt)")
        if path:
            f = open(path, "w")
            for object_ in self.objects:
                f.write("{} {} ".format(object_.get_name(), object_.get_class()))
                if self.height is not None and self.angleSlope is not None and self.angleVert is not None and self.angleHoris is not None:
                    f.write("; ".join(
                        ["({}, {})".format(*self.search_dx_dy(i[0], i[1])) for i in object_.get_trajectory()]))
                else:
                    f.write("; ".join(["({}, {})".format(i[0], i[1]) for i in object_.get_trajectory()]))
                f.write("\n")
            f.close()

    def export_txt(self):
        self._export_txt(None)

    def _save_txt_lat_lon(self, path):
        if self.lat is None and self.lon is None and self.angleDirection:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Экспортировать объекты", "", "Текстовые файлы (*.txt)")
        if path:
            f = open(path, "w")
            for object_ in self.objects:
                f.write("{} {} ".format(object_.get_name(), object_.get_class()))

                if self.pos is None:
                    f.write("; ".join(["({}, {})".format(
                        *self.search_lat_lon(self.lat, self.lon, *self.search_dx_dy(i[0], i[1]), self.angleDirection))
                        for i in object_.get_trajectory()]))
                else:
                    f.write("; ".join(["({}, {})".format(
                        *self.search_from_point(i[0], i[1], self.pos[0], self.pos[1], self.lat, self.lon))
                        for i in object_.get_trajectory()]))
                f.write("\n")
            f.close()

    def save_txt_lat_lon(self):
        self._save_txt_lat_lon(None)

    def save_video(self):
        self._save_video(None)

    def save_video_txt(self):
        self._save_video("txt")

    def save_video_txt_lat_lon(self):
        self._save_video("txt_lat_lon")

    def _save_video(self, info_format):
        if not self.video_path:
            return
        video_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "", "Видеофайлы (*.mp4)")
        if video_path:
            self.clear_info()
            video_capture = cv2.VideoCapture(self.video_path)
            ret, frame = video_capture.read()
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),
                                           video_capture.get(cv2.CAP_PROP_FPS), (width, height))
            while ret:
                processed_frame = cv2.cvtColor(self.process_frame(frame), cv2.COLOR_BGR2RGB)
                video_writer.write(processed_frame)
                ret, frame = video_capture.read()
            video_capture.release()
            if info_format == "txt":
                path = video_path[:-3] + "txt"
                self._export_txt(path)
            if info_format == "txt_lat_lon":
                path = video_path[:-3] + "txt"
                self._save_txt_lat_lon(path)

    def search_lat_lon(self, lat1, lon1, dx, dy, angle):
        # Радиус Земли в метрах
        R = 6378137

        # Конвертация смещения из метров в градусы
        delta_lat = dy / R * (180 / math.pi)
        delta_lon = dx / (R * math.cos(math.radians(lat1))) * (180 / math.pi)

        # Учет угла
        angle_rad = math.radians(angle)
        delta_lat_adj = delta_lat * math.cos(angle_rad) - delta_lon * math.sin(angle_rad)
        delta_lon_adj = delta_lat * math.sin(angle_rad) + delta_lon * math.cos(angle_rad)

        # Координаты второго объекта
        lat2 = lat1 + delta_lat_adj
        lon2 = lon1 + delta_lon_adj
        return lat2, lon2

    def search_from_point(self, x_obj, y_obj, x_point, y_point, lat_point, lon_point):
        """Поиск координат относительно точки"""
        dx_point, dy_point = self.search_dx_dy(x_point, y_point)
        lat_camera, lot_camera = self.search_lat_lon(lat_point, lon_point, dx_point, dy_point,
                                                     self.angleDirection - 180)
        dx_obj, dy_obj = self.search_dx_dy(x_obj, y_obj)
        return self.search_lat_lon(lat_camera, lot_camera, dx_obj, dy_obj, self.angleDirection)

    def search_dx_dy(self, x, y):
        """Через сферическую систему координат"""
        phi = math.radians(90 - math.atan(2 * x - 1) / math.radians(45) * self.angleHoris / 2)
        theta_ = math.radians(math.atan(2 * y - 1) / math.radians(45) * self.angleVert / 2 + self.angleSlope + 90)
        r = - self.height / math.cos(theta_)
        dx = r * math.sin(theta_) * math.cos(phi)
        dy = r * math.sin(theta_) * math.sin(phi)
        return dx, dy


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
