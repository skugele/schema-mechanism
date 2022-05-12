import logging
from collections import Callable

SUPPRESS_PERFORMANCE = True
SUPPRESS_STRING_OUTPUT = True

logger = logging.getLogger('test')


def disable_test(test_case: Callable) -> Callable:
    def _test_wrapper(*_args, **_kwargs):
        logger.warning(f'Suppressing test case: {test_case}')

    return _test_wrapper


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
