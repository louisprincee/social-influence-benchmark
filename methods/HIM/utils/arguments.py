import yaml


class Args:
    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            self._set_attr(key, value)

    def _set_attr(self, key, value):
        self._data[key] = value
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._set_attr(key, value)

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._set_attr(key, value)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return self.__str__()

    def remove_key(self, key):
        """Remove a specific key from Args."""
        if key in self._data:
            del self._data[key]
            object.__delattr__(self, key)
        else:
            raise KeyError(f"Key '{key}' does not exist in Args.")