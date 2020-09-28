import numpy.ma as ma
import datetime


def get_current_time():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def rpg_seconds2date(time_stamp: int, date_only: bool = False) -> list:
    """Convert RPG timestamp to UTC date + time.

    Args:
        time_stamp (int): RPG timestamp.
        date_only (bool): If true, return only date (no time).

    Returns:
        list: UTC date + optionally time in format ['YYYY', 'MM', 'DD', 'hh', 'min', 'sec']

    """
    epoch = datetime.datetime(2001, 1, 1, 0, 0).timestamp()
    time_stamp += epoch
    date_and_time = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y %m %d %H %M %S').split()
    if date_only:
        return date_and_time[:3]
    return date_and_time


def get_rpg_file_type(header):
    """Find level and version of RPG cloud radar binary file.

    Args:
        header (dict): Header of the radar file containing *file_code* key.

    Returns:
        tuple: 2-element tuple containing Level (0 or 1) and Version (2, 3 or 4).

    Raises:
        RuntimeError: Unknown file type.

    """
    file_code = header['FileCode']
    if file_code == 789346:
        return 0, 2
    elif file_code == 889346:
        return 0, 3
    elif file_code == 789347:
        return 1, 2
    elif file_code == 889347:
        return 1, 3
    elif file_code == 889348:
        return 1, 4
    raise RuntimeError('Unknown RPG binary file.')


def isscalar(array):
    """Tests if input is scalar.

    By "scalar" we mean that array has a single value.

    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True

    """
    arr = ma.array(array)
    if not hasattr(arr, '__len__') or arr.shape == () or len(arr) == 1:
        return True
    return False
