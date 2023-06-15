#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  benchmarks.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

import pytest


def my_function():
    # Codice da testare
    pass


def test_my_function(benchmark):
    # Eseguire il benchmark della funzione
    result = benchmark(my_function)


# Eseguire il test con pytest

pytest.main()
