import sys
import gui
import backend
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

# Import necessary modules and backend functions

class WorkerThread(QThread):
    """
    Background thread to run the stock prediction pipeline without freezing the GUI.
    """
    finished = pyqtSignal(str)   # emits the path to the generated plot
    progress = pyqtSignal(str)   # emits status-text updates

    def __init__(self, ticker: str, parent=None):
        # Initialize the thread with the stock ticker symbol
        super().__init__(parent)
        self.ticker = ticker.strip().upper()

    # ------------------------------------------------------------------ #
    # -- THREAD ENTRY POINT --                                           #
    # ------------------------------------------------------------------ #
    def run(self):
        """
        Main entry point for the thread. Executes the data loading, training, and forecasting pipeline.
        """
        try:
            # Load configuration and data, preprocess, train model, and predict future prices
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
            # Handle any errors that occur during the pipeline execution
            self.progress.emit(f"Error: {err!s}")


# ---------------------------------------------------------------------- #
# GUI bootstrap                                                          #
# ---------------------------------------------------------------------- #
def main():
    # Create the application and main window
    app = QApplication(sys.argv)
    win = gui.MainWindow()

    # ---------- helpers wired to buttons / signals ---------- #
    # Helper function to start the forecasting process
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
        # Display the generated plot and re-enable the search button
        win.display_stock_graph(plot_path)
        win.search_button.setEnabled(True)

    # ---------- connect events ---------- #
    # Connect GUI events to their respective handlers
    win.search_button.clicked.connect(start_forecast)
    win.search_entry.returnPressed.connect(start_forecast)
    win.back_button.clicked.connect(win.go_back_to_search)

    # Show the main window and start the application event loop
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()