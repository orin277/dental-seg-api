import logging
import os
from app.core.log_formatters import ColoredFormatter, UvicornJsonFormatter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "[%(asctime)s] %(module)25s:%(lineno)-3d %(levelname)-8s - %(message)s",
            "datefmt": '%Y-%m-%d %H:%M:%S'
        },
        "server_stdout": {
            "()": ColoredFormatter,
            "format": "[%(asctime)s] %(levelname)-8s - %(message)s",
            "datefmt": '%Y-%m-%d %H:%M:%S'
        },
        "server_json": {
            "()": UvicornJsonFormatter,
            "fmt": "[%(asctime)s] %(levelname)-8s - %(message)s",
            "datefmt": '%Y-%m-%d %H:%M:%S'
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "json",
            "level": "DEBUG",
            "filename": os.path.join(LOG_DIR, "app.json"),
            "encoding": "utf-8"
        },
        "server_stdout": {
            "class": "logging.StreamHandler",
            "formatter": "server_stdout",
            "level": "INFO",
            "stream": "ext://sys.stdout"
        },
        "server_file": {
            "class": "logging.FileHandler",
            "formatter": "server_json",
            "level": "INFO",
            "filename": os.path.join(LOG_DIR, "server.json"),
            "encoding": "utf-8"
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["server_stdout", "server_file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["server_stdout", "server_file"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["server_stdout", "server_file"], "level": "INFO", "propagate": False},
        "uvicorn.protocols": {"level": "WARNING"},
        "app": {"handlers": ["file"], "level": "DEBUG"},
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)


class Message:
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)

class BraceStyleAdapter(logging.LoggerAdapter):
    def log(self, level, msg, /, *args, stacklevel=1, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, Message(msg, args), **kwargs,
                            stacklevel=stacklevel+1)


def get_logger(name: str):
    return BraceStyleAdapter(logging.getLogger(name))