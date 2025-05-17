import sys
import gui
import backend
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    """
    Background thread to run the stock prediction pipeline without freezing the GUI.
    """
    finished = pyqtSignal(str)   # Signal emitted when prediction is complete with path to plot
    progress = pyqtSignal(str)   # Signal for status updates during processing

    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker.strip().upper()

    def run(self):
        """
        Main entry point for the thread. Executes the data loading, training, and forecasting pipeline.
        """
        try:
            # Configure and run the forecasting pipeline
            cfg = backend.Config()
            
            # Step 1: Load historical stock data
            self.progress.emit("Loading data from Stooq…")
            data = backend.load_stock_data(self.ticker)

            # Step 2: Preprocess and scale the data
            self.progress.emit("Pre-processing…")
            scaled, scaler = backend.preprocess_data(data, cfg)
            last_seq = scaled[-cfg.time_step :]

            # Step 3: Train the deep learning model
            self.progress.emit("Training deep-learning model…")
            model, _ = backend.train_model(scaled, cfg)

            # Step 4: Generate price predictions
            self.progress.emit(f"Predicting the next {cfg.future_steps} trading days…")
            preds = backend.predict_future(model, last_seq, scaler, cfg)

            # Step 5: Create visualization
            self.progress.emit("Rendering chart…")
            plot_path = backend.plot_forecast(data, preds, cfg)

            # Signal completion with path to the generated chart
            self.finished.emit(plot_path)

        except Exception as err:
            # Report any errors to the UI
            self.progress.emit(f"Error: {err!s}")


def main():
    """
    Application entry point. Sets up the GUI and connects signals to handlers.
    """
    # Initialize application and main window
    app = QApplication(sys.argv)
    win = gui.MainWindow()

    # Handler to start the forecasting process
    def start_forecast():
        ticker = win.search_entry.text().strip()
        if not ticker:
            win.display_status_message("Please enter a valid ticker symbol.")
            return

        win.display_status_message("Initialising…")
        win.search_button.setEnabled(False)

        # Create and start worker thread for background processing
        worker = WorkerThread(ticker)
        worker.progress.connect(win.display_status_message)
        worker.finished.connect(on_done)
        worker.start()

        # Keep reference to prevent garbage collection
        win.thread = worker

    # Handler for when prediction completes
    def on_done(plot_path: str):
        win.display_stock_graph(plot_path)
        win.search_button.setEnabled(True)

    # Connect UI events to handlers
    win.search_button.clicked.connect(start_forecast)
    win.search_entry.returnPressed.connect(start_forecast)
    win.back_button.clicked.connect(win.go_back_to_search)

    # Start the application
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()