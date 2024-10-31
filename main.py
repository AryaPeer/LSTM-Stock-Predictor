# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from backend import predict_stock

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    
    # Connect search button to search action
    def on_search():
        ticker = main_window.search_entry.text()
        plot_path = predict_stock(ticker)  # Call backend function
        main_window.display_stock_graph(plot_path)  # Update GUI with result

    main_window.search_button.clicked.connect(on_search)
    
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
