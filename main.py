import sys

import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMenuBar, QAction, \
    QSlider, QActionGroup, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QPixmap
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
        self.is_stopped = False
        self.current_frame = None
        self.start_frame = 0
        self.model = None
        self.prev_objects = []
        self.objects = []
        self.labels = []

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.play_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderReleased.connect(self.play_video)
        self.slider.sliderMoved.connect(self.set_frame)
        self.slider.sliderPressed.connect(self.stop_video)

        self.play_button.clicked.connect(self.play_video)
        self.stop_button.clicked.connect(self.stop_video)

        detections_layout = QVBoxLayout()

        scroll = QScrollArea()
        detections_layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollContent = QWidget()

        self.scrollLayout = QVBoxLayout()
        scrollContent.setLayout(self.scrollLayout)

        scroll.setWidget(scrollContent)

        self.scrollLayout.setAlignment(Qt.AlignTop)
        scroll.setMinimumSize(200, 20)
        scroll.setMaximumSize(200, 3000)

        video_layout = QHBoxLayout()
        video_layout.addLayout(detections_layout)
        video_layout.addWidget(self.video_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.play_button)
        main_layout.addWidget(self.stop_button)
        main_layout.addWidget(self.slider)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.timeout.connect(self.update_slider_position)

        self.create_menu()

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
        export_txt = QAction(".txt", self)
        export_txt.triggered.connect(self.export_txt)
        export_objects.addAction(export_txt)
        save_video = menu_bar.addMenu("Сохранить")
        save_action = QAction("Сохранить видео", self)
        save_action.triggered.connect(self.save_video)
        save_video.addAction(save_action)
        save_with_txt_action = QAction("Сохранить с txt", self)
        save_with_txt_action.triggered.connect(self.save_video_txt)
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
            self.current_frame = None
        self.is_stopped = True

    def play_video(self):
        if self.video_path and not self.is_playing:
            if self.is_stopped:
                self.video_capture = cv2.VideoCapture(self.video_path)
                self.is_stopped = False
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.is_playing = True
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(fps))

    def update_video(self):
        if self.current_frame is None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
            else:
                self.stop_video()
                return

        processed_frame = self.process_frame(self.current_frame)
        self.update_labels()
        self.display_frame(processed_frame)

        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
        else:
            self.stop_video()

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
            lab.setText("{}\n{}\n{}\n{}\n\n".format(object_.get_name(), object_.get_class(),
                                                    object_.get_last_position()[0], object_.get_last_position()[1]))
            lab.setStyleSheet("font-size: 16px")
            lab.setMinimumSize(200, 80)
            lab.setAlignment(Qt.AlignTop)
            self.labels.append(lab)
            self.scrollLayout.addWidget(lab)

    def export_txt(self):
        path, _ = QFileDialog.getSaveFileName(self, "Экспортировать объекты", "", "Текстовые файлы (*.txt)")
        if path:
            f = open(path, "w")
            for object_ in self.objects:
                f.write("{} {} ".format(object_.get_name(), object_.get_class()))
                f.write("; ".join(["({}, {})".format(i[0], i[1]) for i in object_.get_trajectory()]))
                f.write("\n")
            f.close()

    def save_video(self):
        self._save_video(None)

    def save_video_txt(self):
        self._save_video("txt")

    def _save_video(self, info_format):
        if not self.video_path:
            return
        video_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "", "Видеофайлы (*.mp4)")
        if video_path:
            self.clear_info()
            video_capture = cv2.VideoCapture(self.video_path)
            ret, frame = video_capture.read()
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter.fourcc('m','p','4','v'), video_capture.get(cv2.CAP_PROP_FPS), (width, height))
            while ret:
                processed_frame = cv2.cvtColor(self.process_frame(frame), cv2.COLOR_BGR2RGB)
                video_writer.write(processed_frame)
                ret, frame = video_capture.read()
            video_capture.release()
            if info_format == "txt":
                path = video_path[:-3] + "txt"
                f = open(path, "w")
                for object_ in self.objects:
                    f.write("{} {} ".format(object_.get_name(), object_.get_class()))
                    f.write("; ".join(["({}, {})".format(i[0], i[1]) for i in object_.get_trajectory()]))
                    f.write("\n")
                f.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
