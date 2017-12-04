"""Shared utility functions."""

import types


def inherit_docstrings(cls):
    """Class decorator to inherit parent class docstrings if not specified.

    Taken from https://stackoverflow.com/a/38601305/4798943.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       doc=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls
