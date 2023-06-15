#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  custom_logger.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-14
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

def INFO(message):
    """
    Function used to print a log message which represents a basic info.

    Args:
        message (str): the input message.
    """

    print("\033[1m\033[32mINFO\033[0m: {}".format(message))


def WARNING(message):
    """
    Function used to print a log message which represents a warning.

    Args:
        message (str): the input message.
    """

    print("\033[1m\033[33mWARNING\033[0m: {}".format(message))


def ERROR(message):
    """
    Function used to print a log message which represents an error.

    Args:
        message (str): the input message.
    """

    print("\033[1m\033[31mERROR\033[0m: {}".format(message))


def RESULT(message):
    """
    Function used to print a log message which represents a result.

    Args:
        message (str): the input message.
    """

    print("\033[1m\033[36mRESULT\033[0m: {}".format(message))
