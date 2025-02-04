import sys
import gui
import backend
import pandas as pd
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    """
    Background worker thread that handles the entire data-loading,
    training, prediction, and backtesting sequence.
    Emits progress messages and final results back to the UI.
    """
    finished = pyqtSignal(str, pd.DataFrame)
    progress = pyqtSignal(str)
    
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
    
    def run(self):
        """
        Runs the workflow in a thread:
        1) Load & preprocess data
        2) Perform cross-validation
        3) Train final model
        4) Generate future predictions
        5) Backtest strategy
        6) Plot results
        7) Emit final plot path and results
        """
        try:
            # Step 1: Load data
            self.progress.emit("Loading data...")
            config = backend.EnhancedConfig()
            data = backend.load_stock_data(self.ticker)
            
            # Step 2: Preprocess data
            self.progress.emit("Preprocessing data...")
            scaled_data, scaler = backend.preprocess_data(data, 30)
            
            # Step 3: Cross-validation on multiple splits
            cv_splits = backend.time_series_cv_split(scaled_data, config)
            for i, (train_data, val_data) in enumerate(cv_splits, 1):
                self.progress.emit(f"Training fold {i}/{config.n_splits}...")
                model, metrics = backend.evaluate_fold(train_data, val_data, config, 30)
            
            # Step 4: Train final model with all data
            self.progress.emit("Training final model...")
            final_model, val_loss = backend.build_and_train_model(scaled_data, config, 30)
            
            # Step 5: Predict future prices
            self.progress.emit("Generating predictions...")
            last_sequence = scaled_data[-config.time_step:]
            last_actual_price = data['Close'].values[-1]
            predictions, confidence_intervals = backend.predict_future_prices(
                final_model, 
                last_sequence, 
                scaler, 
                30, 
                last_actual_price 
            )
            
            # Step 6: Run strategy backtest
            self.progress.emit("Running strategy backtesting...")
            backtest_results = backend.backtest_strategy(
                data, 
                predictions, 
                confidence_intervals, 
                config
            )
            
            # Step 7: Plot predictions and emit final results
            plot_path = backend.plot_predictions(data, predictions, 30)
            self.finished.emit(plot_path, backtest_results)
            
        except Exception as e:
            # Report errors to the UI via the progress signal
            self.progress.emit(f"Error: {str(e)}")
            raise

def main():
    """
    Main entry point for the PyQt application.
    Sets up the UI and connects signals/slots.
    """
    app = QApplication(sys.argv)
    main_window = gui.MainWindow()
    
    def on_search():
        """
        Triggered when 'Search' is clicked or Enter is pressed.
        Starts the worker thread if a valid ticker is provided.
        """
        ticker = main_window.search_entry.text().strip().upper()
        if ticker:
            main_window.display_status_message("Initializing...")
            main_window.search_button.setEnabled(False)  # Prevent multiple clicks
            worker = WorkerThread(ticker)
            
            # Connect worker signals to UI functions
            worker.finished.connect(on_prediction_finished)
            worker.progress.connect(main_window.display_status_message)
            worker.start()
            
            # Keep a reference to avoid garbage collection
            main_window.thread = worker
        else:
            main_window.display_status_message("Please enter a valid ticker symbol.")
    
    def on_prediction_finished(plot_path, backtest_results):
        """
        Called when the worker thread finishes.
        Receives the path to the plot image and the backtest DataFrame.
        Updates the UI with metrics and the plot.
        """
        main_window.search_button.setEnabled(True)
        
        # Calculate final performance metrics
        final_return = backtest_results['Cumulative_Return'].iloc[-1]
        final_sharpe = backtest_results['Rolling_Sharpe'].iloc[-1]
        avg_position = backtest_results['Position'].abs().mean()
        
        # Construct status message
        status_message = (
            f"Prediction complete!\n"
            f"Expected Return: {final_return:.1%}\n"
            f"Sharpe Ratio: {final_sharpe:.2f}\n"
            f"Avg Position: {avg_position:.2f}"
        )
        
        main_window.display_status_message(status_message)
        main_window.display_stock_graph(plot_path)
    
    def on_back():
        """Navigates back to the search screen."""
        main_window.go_back_to_search()
    
    # Connect UI elements to functions
    main_window.search_button.clicked.connect(on_search)
    main_window.search_entry.returnPressed.connect(on_search)
    main_window.back_button.clicked.connect(on_back)
    
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()