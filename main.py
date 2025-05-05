import sys
import os
import torch
import clip
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, QScrollArea
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class SemanticImageSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Семантический поиск изображений")
        self.resize(900, 600)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.image_folder = None
        self.image_paths = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Введите текстовое описание...")
        layout.addWidget(self.query_input)

        btn_layout = QHBoxLayout()

        self.load_folder_btn = QPushButton("Выбрать папку с изображениями")
        self.load_folder_btn.clicked.connect(self.select_image_folder)
        btn_layout.addWidget(self.load_folder_btn)

        self.search_btn = QPushButton("Поиск")
        self.search_btn.clicked.connect(self.search_images)
        btn_layout.addWidget(self.search_btn)

        layout.addLayout(btn_layout)

        self.result_area = QScrollArea()
        self.result_widget = QWidget()
        self.result_layout = QHBoxLayout()
        self.result_widget.setLayout(self.result_layout)
        self.result_area.setWidgetResizable(True)
        self.result_area.setWidget(self.result_widget)

        layout.addWidget(self.result_area)
        self.setLayout(layout)

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder:
            self.image_folder = folder
            self.image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]

    def search_images(self):
        text = self.query_input.text().strip()
        if not text or not self.image_paths:
            return

        self.result_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().setParent(None)

        image_tensors = []
        valid_paths = []

        for path in self.image_paths:
            try:
                img = self.preprocess(Image.open(path).convert("RGB"))
                image_tensors.append(img)
                valid_paths.append(path)
            except Exception:
                continue

        if not image_tensors:
            return

        image_input = torch.stack(image_tensors).to(self.device)
        text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).squeeze()
            top_indices = similarity.topk(5).indices

        for idx in top_indices:
            path = valid_paths[idx]
            pixmap = QPixmap(path).scaledToWidth(150, Qt.TransformationMode.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            self.result_layout.addWidget(label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SemanticImageSearch()
    window.show()
    sys.exit(app.exec())
