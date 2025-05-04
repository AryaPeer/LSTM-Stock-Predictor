import sys
import gui
import backend
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    """
    Background thread to run the stock prediction pipeline without freezing the GUI.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker.strip().upper()

    def run(self):
        """
        Main entry point for the thread. Executes the data loading, training, and forecasting pipeline.
        """
        try:
            cfg = backend.Config()
            
            self.progress.emit("Loading data from Stooq…")
            data = backend.load_stock_data(self.ticker)

            self.progress.emit("Pre-processing…")
            scaled, scaler = backend.preprocess_data(data, cfg)
            last_seq = scaled[-cfg.time_step :]

            self.progress.emit("Training deep-learning model…")
            model, _ = backend.train_model(scaled, cfg)

            self.progress.emit(f"Predicting the next {cfg.future_steps} trading days…")
            preds = backend.predict_future(model, last_seq, scaler, cfg)

            self.progress.emit("Rendering chart…")
            plot_path = backend.plot_forecast(data, preds, cfg)

            self.finished.emit(plot_path)

        except Exception as err:
            self.progress.emit(f"Error: {err!s}")


def main():
    """
    Application entry point. Sets up the GUI and connects signals to handlers.
    """
    app = QApplication(sys.argv)
    win = gui.MainWindow()

    def start_forecast():
        ticker = win.search_entry.text().strip()
        if not ticker:
            win.display_status_message("Please enter a valid ticker symbol.")
            return

        win.display_status_message("Initialising…")
        win.search_button.setEnabled(False)

        worker = WorkerThread(ticker)
        worker.progress.connect(win.display_status_message)
        worker.finished.connect(on_done)
        worker.start()

        win.thread = worker # Keep reference

    def on_done(plot_path: str):
        win.display_stock_graph(plot_path)
        win.search_button.setEnabled(True)

    win.search_button.clicked.connect(start_forecast)
    win.search_entry.returnPressed.connect(start_forecast)
    win.back_button.clicked.connect(win.go_back_to_search)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()