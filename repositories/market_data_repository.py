import pandas as pd
import pandas_datareader.data as web
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class MarketDataRepository:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache = {}

    def get_stock_data(self,
                       ticker: str,
                       start_date: str = None,
                       end_date: str = None,
                       use_cache: bool = True) -> pd.DataFrame:

        # Default dates
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"{ticker}_{start_date}_{end_date}"

        # Check memory cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Returning {ticker} from memory cache")
            return self._cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() < 3600:  # 1 hour cache
                logger.debug(f"Loading {ticker} from disk cache")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._cache[cache_key] = data
                    return data

        # Fetch from data source
        logger.info(f"Fetching {ticker} from data source")
        try:
            df = web.DataReader(ticker, 'stooq', start_date, end_date)
            if df.empty:
                raise ValueError(f"No data available for {ticker}")

            df = df.sort_index()
            df.index = pd.to_datetime(df.index)

            # Cache the data
            self._cache[cache_key] = df
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            raise

    def get_market_index(self, index: str = "^SPX") -> pd.DataFrame:
        # Get S&P 500 or other market index data
        return self.get_stock_data(index.lower())

    def get_sector_data(self, sector: str) -> Dict[str, pd.DataFrame]:
        # Map sectors to ETFs
        sector_etfs = {
            'technology': 'XLK',
            'healthcare': 'XLV',
            'financials': 'XLF',
            'energy': 'XLE',
            'consumer': 'XLY',
            'industrials': 'XLI',
            'utilities': 'XLU',
            'realestate': 'XLRE',
            'materials': 'XLB'
        }

        etf = sector_etfs.get(sector.lower())
        if etf:
            return {sector: self.get_stock_data(etf)}
        return {}

    def clear_cache(self, ticker: Optional[str] = None):
        if ticker:
            # Clear specific ticker cache
            for key in list(self._cache.keys()):
                if ticker in key:
                    del self._cache[key]

            for cache_file in self.cache_dir.glob(f"{ticker}_*.pkl"):
                cache_file.unlink()
        else:
            # Clear all cache
            self._cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
