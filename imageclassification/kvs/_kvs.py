import datetime
import pickle


class GlobalKVS(object):
    _instance = None
    _d = dict()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalKVS, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def update(self, tag, value, dtype=None):
        """
        Updates the internal state of the logger.
        Parameters
        ----------
        tag : str
            Tag, of the variable, which we log.
        value : object
            The value to be logged
        dtype :
            Container which is used to store the values under the tag
        Returns
        -------
        """
        if tag not in self._d:
            if dtype is None:
                self._d[tag] = (value, str(datetime.datetime.now()))
            else:
                self._d[tag] = dtype()
        else:
            if isinstance(self._d[tag], list):
                self._d[tag].append((value, str(datetime.datetime.now())))
            elif isinstance(self._d[tag], dict):
                self._d[tag].update((value, str(datetime.datetime.now())))
            else:
                self._d[tag] = (value, str(datetime.datetime.now()))

    def __getitem__(self, tag):
        if not isinstance(self._d[tag], (list, dict)):
            return self._d[tag][0]
        else:
            return self._d[tag]

    def tag_ts(self, tag):
        return self._d[tag]

    def save_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._d, f)
