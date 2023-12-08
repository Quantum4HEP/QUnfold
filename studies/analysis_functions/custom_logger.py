# ---------------------- Metadata ----------------------
#
# File name:  CustomLogger.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-07
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import logging


class CustomColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages based on the log level.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
        "RESET": "\033[0m",  # Reset color
    }

    def format(self, record):
        """
        Format a log record by adding colors to the log level.

        Args:
            record (LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message with colors.
        """

        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = (
            f"{log_color}{record.levelname}\033[0m"  # Reset color after levelname
        )
        return super().format(record)


def get_custom_logger(name):
    """
    Get a custom logger with a colored formatter.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: An instance of the logger configured with the colored formatter.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(CustomColoredFormatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
