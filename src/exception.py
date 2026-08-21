import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error message including filename, line number, and error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{str(error)}]"
        )
    else:
        error_message = f"Error: [{str(error)}]"
    return error_message


class PlantDiseaseException(Exception):
    """
    Custom exception class for Plant Disease Classifier project.
    """

    def __init__(self, error_message: str, error_detail: Optional[sys] = sys):
        super().__init__(error_message)
        if error_detail is not None and hasattr(error_detail, "exc_info"):
            self.error_message = error_message_detail(error_message, error_detail=error_detail)
        else:
            self.error_message = str(error_message)

    def __str__(self) -> str:
        return self.error_message
