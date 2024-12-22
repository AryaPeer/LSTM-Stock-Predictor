from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QStackedLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.base_size = QSize(400, 50)
        self.result_size = QSize(800, 500)  # Define result window size
        self.initUI()
      
    def initUI(self):
        self.setWindowTitle("Stock Predictor")
        self.resize(self.base_size) 

        self.stacked_layout = QStackedLayout()
        self.search_widget = QWidget()
        self.search_layout = QVBoxLayout()
        
        self.search_entry = QLineEdit(self)
        self.search_entry.setPlaceholderText("Enter ticker symbol")
        self.search_layout.addWidget(self.search_entry, alignment=Qt.AlignTop)
        
        self.search_button = QPushButton("Search", self)
        self.search_layout.addWidget(self.search_button, alignment=Qt.AlignTop)

        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.search_layout.addWidget(self.status_label, alignment=Qt.AlignTop)
        
        self.search_widget.setLayout(self.search_layout)
        
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout()
        
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setScaledContents(True)
        self.result_layout.addWidget(self.result_label, stretch=1)

        self.back_button = QPushButton("Back", self)
        self.result_layout.addWidget(self.back_button, alignment=Qt.AlignBottom)
        
        self.result_widget.setLayout(self.result_layout)
        
        self.stacked_layout.addWidget(self.search_widget)
        self.stacked_layout.addWidget(self.result_widget)
        
        self.setLayout(self.stacked_layout)
    
    def display_status_message(self, message):
        self.status_label.setText(message)
    
    def display_stock_graph(self, plot_path):
        pixmap = QPixmap(plot_path)

        scaled_pixmap = pixmap.scaled(
            self.result_size.width() - 40, 
            self.result_size.height() - 100, 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        self.resize(self.result_size)
        self.stacked_layout.setCurrentWidget(self.result_widget)

    def go_back_to_search(self):
        self.status_label.setText("")
        self.search_entry.clear()
        self.stacked_layout.setCurrentWidget(self.search_widget)
        self.resize(self.base_size)
