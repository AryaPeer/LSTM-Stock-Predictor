from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QStackedLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize

class MainWindow(QWidget):
    """
    Main window for the stock prediction app. Provides a search view for inputting a ticker symbol
    and a result view for displaying the stock price plot.
    """
    def __init__(self):
        # Initialize the main window and set up the UI
        super().__init__()
        # Set default window sizes for both views
        self.base_size = QSize(400, 50)
        self.result_size = QSize(800, 500)
        
        self.initUI()
      
    def initUI(self):
        """
        Set up the primary UI layout, including the search view and result view.
        """
        # Configure window properties
        self.setWindowTitle("Stock Predictor")
        self.resize(self.base_size)  # Start with compact size
        
        # Create stacked layout to switch between views
        self.stacked_layout = QStackedLayout()
        
        # Search view setup
        self.search_widget = QWidget()
        self.search_layout = QVBoxLayout()
        
        # Ticker input field
        self.search_entry = QLineEdit(self)
        self.search_entry.setPlaceholderText("Enter ticker symbol")
        self.search_layout.addWidget(self.search_entry, alignment=Qt.AlignTop)
        
        # Search button
        self.search_button = QPushButton("Search", self)
        self.search_layout.addWidget(self.search_button, alignment=Qt.AlignTop)

        # Status display area
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.search_layout.addWidget(self.status_label, alignment=Qt.AlignTop)
        
        self.search_widget.setLayout(self.search_layout)
        
        # Results view setup
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout()
        
        # Image display area for the forecast chart
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setScaledContents(True)
        self.result_layout.addWidget(self.result_label, stretch=1)

        # Back button to return to search
        self.back_button = QPushButton("Back", self)
        self.result_layout.addWidget(self.back_button, alignment=Qt.AlignBottom)
        
        self.result_widget.setLayout(self.result_layout)
        
        # Add both views to stacked layout
        self.stacked_layout.addWidget(self.search_widget)
        self.stacked_layout.addWidget(self.result_widget)
        
        self.setLayout(self.stacked_layout)
    
    def display_status_message(self, message):
        """
        Update the status label in the search view with a given message.
        
        Args:
            message: Text to display in the status label
        """
        self.status_label.setText(message)
    
    def display_stock_graph(self, plot_path):
        """
        Display the stock price chart in the result view.
        
        Args:
            plot_path: File path to the saved plot image
        """
        # Load and scale the image for display
        pixmap = QPixmap(plot_path)

        # Scale the image to fit nicely in the result view
        scaled_pixmap = pixmap.scaled(
            self.result_size.width() - 40, 
            self.result_size.height() - 100, 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)

        # Resize window and switch to result view
        self.resize(self.result_size)
        self.stacked_layout.setCurrentWidget(self.result_widget)

    def go_back_to_search(self):
        """
        Clear the result view and return to the search view.
        """
        # Reset UI elements and switch back to search
        self.status_label.setText("")
        self.search_entry.clear()
        self.result_label.clear()
        self.stacked_layout.setCurrentWidget(self.search_widget)
        self.resize(self.base_size)
