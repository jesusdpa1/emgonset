# emgonset/internals.py
import sys


def public_api(obj):
    """
    Decorator to mark a function or class as part of the public API,
    automatically adding it to the module's __all__ list.

    Usage:
        @public_api
        def my_function():
            pass
    """
    module = obj.__module__
    module_obj = sys.modules[module]
    if not hasattr(module_obj, "__all__"):
        module_obj.__all__ = []
    if obj.__name__ not in module_obj.__all__:
        module_obj.__all__.append(obj.__name__)
    return obj
