import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message with script name, line number, and the actual error.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = "Error occurred in Python script [{0}] at line [{1}] with error message: [{2}]".format(
        file_name, line_no, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
