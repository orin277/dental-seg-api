from fastapi import HTTPException, status


class FileNotTransferredException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file was not transferred"
        )


class IncorrectFileTypeException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Incorrect file type"
        )


class BigFileSizeException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The file size is too big"
        )


class FailedToDecodeImageException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decode the image. The file may be corrupted or in an unsupported format"
        )