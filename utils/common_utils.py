from contextlib import contextmanager
import time
from typing import Callable, Optional
from functools import wraps

def set_timer(debug:Optional[bool] = None):
    """
    Decorator to measure time of a function.
    
    Args:    
    - debug: whether to print the time or not
        
    Returns:
    - wrapper: function to measure time
    
    Usage:
    @set_timer(debug=True)
    def test_func():
        time.sleep(1)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs): # 직접 self로 받으면 코드가 작동하지 않음.
            self = args[0] if args else None
            if not self:
                return func(*args, **kwargs)          
            
            method_debug = getattr(self, f"_{func.__name__}_debug", None)
            # 우선순위: 1. 데코레이터의 debug 설정, 2. 메소드의 debug 설정, 3. 클래스의 debug 설정
            effective_debug = debug if debug is not None else (method_debug if method_debug is not None else self.debug)  
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if effective_debug:
                print(f"{self.__name__} / {func.__name__}  elapsed time: {end - start}")
            return result
        return wrapper
    return decorator
