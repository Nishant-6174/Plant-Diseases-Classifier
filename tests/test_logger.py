import os
import glob
import time
from src.logger import logger, LOGS_DIR


def test_logger_creates_log_file():
    """
    Test that logger writes messages to disk log files in the logs/ directory.
    """
    test_msg = "Automated test logger entry - system validation check"
    logger.info(test_msg)
    
    # Flush all handlers to disk
    for handler in logger.handlers:
        handler.flush()

    assert os.path.exists(LOGS_DIR), "Logs directory should exist"
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))
    assert len(log_files) > 0, "At least one log file should be present in logs/"

    # Check that any of the recent log files contains the test message
    found = False
    for log_file in sorted(log_files, key=os.path.getmtime, reverse=True)[:3]:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                if test_msg in content:
                    found = True
                    break
        except Exception:
            continue

    assert found, f"Test log message was not found in recent logs in {LOGS_DIR}"
