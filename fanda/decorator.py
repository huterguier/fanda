import functools

def pipeable(func):
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        func(*args, **kwargs)
        return df
    return wrapper
