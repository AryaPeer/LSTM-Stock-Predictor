import time
import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def validate_ticker(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract ticker from args or kwargs
        ticker = None
        if args:
            ticker = args[0] if isinstance(args[0], str) else None
        if not ticker and 'ticker' in kwargs:
            ticker = kwargs['ticker']

        if ticker:
            # Validate ticker format
            if not ticker.isalnum() or len(ticker) > 5:
                raise ValueError(f"Invalid ticker symbol: {ticker}")

            # Convert to uppercase
            if args:
                args = (ticker.upper(), *args[1:])
            if 'ticker' in kwargs:
                kwargs['ticker'] = ticker.upper()

        return func(*args, **kwargs)
    return wrapper
