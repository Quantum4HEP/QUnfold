#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  conftest.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.


def pytest_addoption(parser):
    """
    Pytest hook to add command-line options.

    This function is called during the command-line option parsing phase of pytest.
    It allows you to add custom command-line options.

    Args:
        parser: The pytest parser object.

    Options:
        --distr: Specify the value for the "distr" option. Default is "normal".
    """
    parser.addoption("--distr", action="store", default="normal")


def pytest_generate_tests(metafunc):
    """
    Pytest hook to generate tests dynamically.

    This function is called for every test. It allows you to generate tests dynamically
    based on command-line arguments or other factors.

    Args:
        metafunc: The pytest metafunc object.

    Fixtures:
        distr: The fixture name that matches the command-line option "distr".

    Example:
        If the command-line option "--distr" is specified, the "distr" fixture will be
        parametrized with the value of the "--distr" option.

        For example, if "--distr=my_distr" is specified, the "distr" fixture will have the
        value ["my_distr"].

        You can then use the "distr" fixture in your tests to access the parameter value.
    """
    option_value = metafunc.config.option.distr
    if "distr" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("distr", [option_value])
