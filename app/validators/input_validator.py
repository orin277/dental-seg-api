from app.exceptions.model import BigFileSizeException, FileNotTransferredException, IncorrectFileTypeException
from app.core.logger import get_logger

logger = get_logger(__name__)

class InputValidator:
    def __init__(self, max_size_mb=10):
        self.max_size_mb = max_size_mb

    def check_file_existence(self, file):
        if file is None:
            logger.error("File not transferred {}", file.filename)
            raise FileNotTransferredException()
        
    def check_image_format(self, file):
        if not file.content_type.startswith("image/"):
            logger.error("Incorrect file type {}", file.filename)
            raise IncorrectFileTypeException()

    def check_file_size(self, contents):
        if len(contents) > self.max_size_mb * 1024 * 1024:
            logger.error("Too big file {}", len(contents))
            raise BigFileSizeException()