from typing import Callable

SUPPRESS_PERFORMANCE = False
SUPPRESS_STRING_OUTPUT = False


def performance_test(test_case: Callable) -> Callable:
    def _perf_test_wrapper(*args, **kwargs):
        if SUPPRESS_PERFORMANCE:
            return
        test_case(*args, **kwargs)

    return _perf_test_wrapper


def string_test(test_case: Callable) -> Callable:
    def _str_test_wrapper(*args, **kwargs):
        if SUPPRESS_STRING_OUTPUT:
            return
        test_case(*args, **kwargs)

    return _str_test_wrapper
