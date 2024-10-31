# gui.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Stock Predictor")
        self.layout = QVBoxLayout()

        # Text entry and search button
        self.search_entry = QLineEdit(self)
        self.search_entry.setPlaceholderText("Enter ticker symbol")
        self.layout.addWidget(self.search_entry, alignment=Qt.AlignTop)
        
        self.search_button = QPushButton("Search", self)
        self.layout.addWidget(self.search_button, alignment=Qt.AlignTop)

        # Placeholder for result image with stretching enabled
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label, stretch=1)  # Allows image to expand
        
        self.setLayout(self.layout)

        # Set resizing policy for the result image
        self.result_label.setScaledContents(True)

    def display_stock_graph(self, plot_path):
        # Load and display the saved matplotlib plot
        self.result_label.setPixmap(QPixmap(plot_path))
