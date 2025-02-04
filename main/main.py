import sys
import gui
import backend
import pandas as pd
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    finished = pyqtSignal(str, pd.DataFrame)
    progress = pyqtSignal(str)
    
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
    
    def run(self):
        try:
            # Step 1: Load data
            self.progress.emit("Loading data...")
            config = backend.EnhancedConfig()
            data = backend.load_stock_data(self.ticker)
            
            # Step 2: Preprocess data
            self.progress.emit("Preprocessing data...")
            scaled_data, scaler = backend.preprocess_data(data, 30) 
            
            # Step 3: Cross-validation
            cv_splits = backend.time_series_cv_split(scaled_data, config)
            for i, (train_data, val_data) in enumerate(cv_splits, 1):
                self.progress.emit(f"Training fold {i}/{config.n_splits}...")
                model, metrics = backend.evaluate_fold(train_data, val_data, config, 30) 
            
            # Step 4: Train final model
            self.progress.emit("Training final model...")
            final_model, val_loss = backend.build_and_train_model(scaled_data, config, 30)
            
            # Step 5: Generate predictions with confidence intervals
            self.progress.emit("Generating predictions...")
            last_sequence = scaled_data[-config.time_step:]
            last_actual_price = data['Close'].values[-1]
            predictions, confidence_intervals = backend.predict_future_prices(
                final_model, last_sequence, scaler, 30, last_actual_price 
            )
            
            # Step 6: Run backtesting
            self.progress.emit("Running strategy backtesting...")
            backtest_results = backend.backtest_strategy(
                data, predictions, confidence_intervals, config
            )
            
            # Step 7: Plot predictions
            plot_path = backend.plot_predictions(data, predictions, 30)  # Added ticker parameter
            
            # Emit both plot path and backtest results
            self.finished.emit(plot_path, backtest_results)
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            raise

def main():
    app = QApplication(sys.argv)
    main_window = gui.MainWindow()
    
    def on_search():
        ticker = main_window.search_entry.text().strip().upper()
        if ticker:
            main_window.display_status_message("Initializing...")
            main_window.search_button.setEnabled(False)
            worker = WorkerThread(ticker)
            worker.finished.connect(on_prediction_finished)
            worker.progress.connect(main_window.display_status_message)
            worker.start()
            main_window.thread = worker
        else:
            main_window.display_status_message("Please enter a valid ticker symbol.")
    
    def on_prediction_finished(plot_path, backtest_results):
        main_window.search_button.setEnabled(True)
        
        # Calculate and display key metrics
        final_return = backtest_results['Cumulative_Return'].iloc[-1]
        final_sharpe = backtest_results['Rolling_Sharpe'].iloc[-1]
        avg_position = backtest_results['Position'].abs().mean()
        
        status_message = (
            f"Prediction complete!\n"
            f"Expected Return: {final_return:.1%}\n"
            f"Sharpe Ratio: {final_sharpe:.2f}\n"
            f"Avg Position: {avg_position:.2f}"
        )
        
        main_window.display_status_message(status_message)
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