import sys
import gui
import backend
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        plot_path = backend.predict_stock(self.ticker)
        self.finished.emit(plot_path)

def main():
    app = QApplication(sys.argv)
    main_window = gui.MainWindow()
    
    def on_search():
        ticker = main_window.search_entry.text().strip()
        if ticker:
            main_window.display_status_message("Training the model, please wait...")
            main_window.search_button.setEnabled(False)
            main_window.thread = WorkerThread(ticker)
            main_window.thread.finished.connect(on_prediction_finished)
            main_window.thread.start()
        else:
            main_window.display_status_message("Please enter a valid ticker symbol.")

    def on_prediction_finished(plot_path):
        main_window.search_button.setEnabled(True)
        main_window.display_status_message("")
        main_window.display_stock_graph(plot_path)

    def on_back():
        main_window.go_back_to_search()

    main_window.search_button.clicked.connect(on_search)
    main_window.search_entry.returnPressed.connect(on_search) 
    main_window.back_button.clicked.connect(on_back)
    
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
