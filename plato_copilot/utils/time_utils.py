import os
import traceback
import time
import inspect
import signal

def timeit(func):
    """
    Decorator function to time a given function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

class Timer:
    def __init__(self, unit="second", verbose=False):
        if unit == "second":
            self.factor = 10 ** 9
        elif unit == "millisecond":
            self.factor = 10 ** 6
        elif unit == "microsecond":
            self.factor = 10 ** 3
        
        self.verbose = verbose
        self.value = None

    def __enter__(self):
        frame = inspect.currentframe()
        self.line_number = frame.f_back.f_lineno
        self.filename = inspect.getframeinfo(frame.f_back).filename
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        elapsed_time = (end_time - self.start_time) / self.factor
        if self.verbose:
            print(f"{elapsed_time} seconds to execute in the block of {self.filename}: {self.line_number}.")

        self.value = elapsed_time

        return None        

    def get_elapsed_time(self):
        return self.value
