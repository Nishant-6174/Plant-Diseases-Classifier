import sys
import pytest
from src.exception import PlantDiseaseException, error_message_detail


def test_custom_exception_formatting():
    """
    Test that PlantDiseaseException captures file name, line number, and message correctly.
    """
    try:
        # Trigger intentional ZeroDivisionError
        _ = 1 / 0
    except Exception as e:
        custom_exc = PlantDiseaseException(e, sys)
        msg = str(custom_exc)
        
        assert "Error occurred in python script" in msg
        assert "line number" in msg
        assert "division by zero" in msg


def test_custom_exception_without_exc_info():
    """
    Test fallback message formatting when sys detail is none.
    """
    err = PlantDiseaseException("Test generic failure", None)
    assert str(err) == "Test generic failure"
