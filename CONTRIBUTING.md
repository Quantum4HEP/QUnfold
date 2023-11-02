# Contributing

## Table of contents

- [How to contribute](#how-to-contribute)
- [How to raise an issue](#how-to-raise-an-issue)
- [How to ask for a feature](#how-to-ask-for-a-feature)
- [Naming conventions](#naming-conventions)

## How to contribute

To contribute:

1. Fork your own copy of the repository.
2. Work on your changes / improvements in the forked repo. Always use `tox` with the corresponding `tox.ini` files and please follow the [conventions](#naming-conventions).
3. **Test** if your changes / improvements are correctly implemented and all the other features are not compromised. If necessary, update test routines of the repository.
4. Finally, make the pull request. The Doxygen page will be updated automatically.

## How to raise an issue

Open an issue using the corresponding [template](https://github.com/JustWhit3/QUnfold/tree/main/.github/ISSUE_TEMPLATE/bug_report.md).

## How to ask for a feature

Open an issue using the corresponding [template](https://github.com/JustWhit3/QUnfold/tree/main/.github/ISSUE_TEMPLATE/feature_request.md).

## Naming conventions

Follow these conventions when writing code:

- Always write a basic heading when writing a new code file, this is useful to know who created a given file. Example:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  __init__.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-06-13
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.
```

- Class names: every word must begin with uppercase (ex: `ThisClass`, `ThisExampleClass`).
- Functions: must use only lowercase letters and each word must be separated by underscore (ex: `my_function`, `function`).
- Variables: must use only lowercase letters and each word must be separated by underscore (ex: `my_var`, `var`).
- Add docstrings in every function or class you create.