import sys
import gui
import backend
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    """
    Runs the complete fetch-train-forecast pipeline in the background
    so the GUI never freezes.
    """
    finished = pyqtSignal(str)   # emits the path to the generated plot
    progress = pyqtSignal(str)   # emits status-text updates

    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker.strip().upper()

    # ------------------------------------------------------------------ #
    # -- THREAD ENTRY POINT --                                           #
    # ------------------------------------------------------------------ #
    def run(self):
        try:
            cfg = backend.Config()                      # new config class
            self.progress.emit("Loading data from Stooq…")
            data = backend.load_stock_data(self.ticker)

            self.progress.emit("Pre-processing…")
            scaled, scaler = backend.preprocess_data(data, cfg)
            last_seq = scaled[-cfg.time_step :]

            self.progress.emit("Training deep-learning model…")
            model, _ = backend.train_model(scaled, cfg)

            self.progress.emit(
                f"Predicting the next {cfg.future_steps} trading days…")
            preds = backend.predict_future(model, last_seq, scaler, cfg)

            self.progress.emit("Rendering chart…")
            plot_path = backend.plot_forecast(data, preds, cfg)

            # hand the plot back to the GUI
            self.finished.emit(plot_path)

        except Exception as err:
            self.progress.emit(f"Error: {err!s}")


# ---------------------------------------------------------------------- #
# GUI bootstrap                                                          #
# ---------------------------------------------------------------------- #
def main():
    app = QApplication(sys.argv)
    win = gui.MainWindow()

    # ---------- helpers wired to buttons / signals ---------- #
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

        win.thread = worker           # keep reference alive

    def on_done(plot_path: str):
        win.display_stock_graph(plot_path)
        win.search_button.setEnabled(True)

    # ---------- connect events ---------- #
    win.search_button.clicked.connect(start_forecast)
    win.search_entry.returnPressed.connect(start_forecast)
    win.back_button.clicked.connect(win.go_back_to_search)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()