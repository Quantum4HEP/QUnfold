#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 14 00:43:00 2023
Author: Gianluca Bianco
"""

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