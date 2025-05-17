from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QStackedLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize

class MainWindow(QWidget):
    """
    Main window for a simple stock prediction app.
    Allows users to input a ticker symbol and then displays
    the resulting stock price plot on a new screen.
    """
    def __init__(self):
        super().__init__()
        # Define default window sizes for the input view and result view
        self.base_size = QSize(400, 50)
        self.result_size = QSize(800, 500)
        
        self.initUI()
      
    def initUI(self):
        """
        Sets up the primary UI layout, including:
        - A stacked layout that toggles between the search view and the result view
        - Search view: input box + search button + status label
        - Result view: label displaying the plot + back button
        """
        self.setWindowTitle("Stock Predictor")
        self.resize(self.base_size)  # Start with base size
        
        # Create a stacked layout to switch between pages
        self.stacked_layout = QStackedLayout()
        
        # -----------------------
        # Search widget setup
        # -----------------------
        self.search_widget = QWidget()
        self.search_layout = QVBoxLayout()
        
        # Entry field for the ticker symbol
        self.search_entry = QLineEdit(self)
        self.search_entry.setPlaceholderText("Enter ticker symbol")
        self.search_layout.addWidget(self.search_entry, alignment=Qt.AlignTop)
        
        # Button to trigger the search/prediction
        self.search_button = QPushButton("Search", self)
        self.search_layout.addWidget(self.search_button, alignment=Qt.AlignTop)

        # Label to display status messages (e.g., errors or "Loading...")
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.search_layout.addWidget(self.status_label, alignment=Qt.AlignTop)
        
        self.search_widget.setLayout(self.search_layout)
        
        # -----------------------
        # Result widget setup
        # -----------------------
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout()
        
        # Label to show the stock graph as a QPixmap
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        # Allows the QPixmap to scale within the label
        self.result_label.setScaledContents(True)
        self.result_layout.addWidget(self.result_label, stretch=1)

        # Button to navigate back to the search screen
        self.back_button = QPushButton("Back", self)
        self.result_layout.addWidget(self.back_button, alignment=Qt.AlignBottom)
        
        self.result_widget.setLayout(self.result_layout)
        
        # -----------------------
        # Add widgets to the stacked layout
        # -----------------------
        self.stacked_layout.addWidget(self.search_widget)
        self.stacked_layout.addWidget(self.result_widget)
        
        self.setLayout(self.stacked_layout)
    
    def display_status_message(self, message):
        """
        Update the status label in the search view.
        
        :param message: Text to display in the status label
        """
        self.status_label.setText(message)
    
    def display_stock_graph(self, plot_path):
        """
        Displays the stock price chart (as a PNG or similar image) in the result view.
        
        :param plot_path: File path to the saved plot image
        """
        pixmap = QPixmap(plot_path)

        # Scale the image to fit nicely in the result widget
        scaled_pixmap = pixmap.scaled(
            self.result_size.width() - 40, 
            self.result_size.height() - 100, 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        
        # Resize the window to accommodate the result view
        self.resize(self.result_size)
        # Switch to the result screen
        self.stacked_layout.setCurrentWidget(self.result_widget)

    def go_back_to_search(self):
        self.status_label.setText("")
        self.search_entry.clear()
        self.result_label.clear()
        self.stacked_layout.setCurrentWidget(self.search_widget)
        self.resize(self.base_size)
