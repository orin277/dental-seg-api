from copy import copy
from logging import Formatter
from pythonjsonlogger import jsonlogger


MAPPING = {
    'DEBUG'   : 37, # white
    'INFO'    : 36, # cyan
    'WARNING' : 33, # yellow
    'ERROR'   : 31, # red
    'CRITICAL': 41, # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'

class ColoredFormatter(Formatter):
    def __init__(self, fmt, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37)
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(PREFIX, seq, levelname, SUFFIX)
        colored_record.levelname = colored_levelname
        return Formatter.format(self, colored_record)
    

class UvicornJsonFormatter(jsonlogger.JsonFormatter):
    def process_log_record(self, record):
        record.pop("color_message", None)
        return super().process_log_record(record)