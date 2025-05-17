from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QStackedLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize

# Import necessary PyQt5 modules for GUI components

class MainWindow(QWidget):
    """
    Main window for the stock prediction app. Provides a search view for inputting a ticker symbol
    and a result view for displaying the stock price plot.
    """
    def __init__(self):
        # Initialize the main window and set up the UI
        super().__init__()
        # Define default window sizes for the input view and result view
        self.base_size = QSize(400, 50)
        self.result_size = QSize(800, 500)
        
        self.initUI()
      
    def initUI(self):
        """
        Set up the primary UI layout, including the search view and result view.
        """
        # Configure the window title and initial size
        self.setWindowTitle("Stock Predictor")
        self.resize(self.base_size)  # Start with base size
        
        # Create a stacked layout to toggle between search and result views
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
        Update the status label in the search view with a given message.
        
        :param message: Text to display in the status label
        """
        # ...existing code...
        self.status_label.setText(message)
    
    def display_stock_graph(self, plot_path):
        """
        Display the stock price chart in the result view.
        
        :param plot_path: File path to the saved plot image
        """
        # Load the plot image and scale it to fit the result view
        pixmap = QPixmap(plot_path)

        # Scale the image to fit nicely in the result widget
        scaled_pixmap = pixmap.scaled(
            self.result_size.width() - 40, 
            self.result_size.height() - 100, 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)

        # Resize the window and switch to the result view
        self.resize(self.result_size)
        # Switch to the result screen
        self.stacked_layout.setCurrentWidget(self.result_widget)

    def go_back_to_search(self):
        """
        Clear the result view and return to the search view.
        """
        # Reset the status label, clear the input field, and switch views
        self.status_label.setText("")
        self.search_entry.clear()
        self.result_label.clear()
        self.stacked_layout.setCurrentWidget(self.search_widget)
        self.resize(self.base_size)
