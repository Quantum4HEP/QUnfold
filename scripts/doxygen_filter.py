import re

def filter(source_file, source_dir, output_dir):
    """
    Filter the source file by converting docstrings to Doxygen-style comments.

    This function reads the source file, identifies docstrings, and converts them into
    Doxygen-style comments. The filtered output is saved in the specified output directory.

    Args:
        source_file (str): The path to the source file.
        source_dir (str): The path to the source directory.
        output_dir (str): The path to the output directory.
    """
    
    with open(source_file, 'r') as f:
        lines = f.readlines()

    doxygen_lines = []
    inside_docstring = False

    for line in lines:
        if re.match(r'\s*\"{3}', line):
            if inside_docstring:
                doxygen_lines.append('///')
            else:
                doxygen_lines.append('/// <')
            inside_docstring = not inside_docstring
        elif inside_docstring:
            doxygen_lines.append('/// ' + line.strip())

    with open(output_dir + '/filtered.py', 'w') as f:
        f.writelines(doxygen_lines)

filter('$file', '$dir', '$outdir')