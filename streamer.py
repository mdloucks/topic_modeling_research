
import json

'''
This file includes custom tools for streaming (reading & writing) large
amounts of data. It's meant to allow us to operate with far less memory.
'''

def line_gen(filename):
    """Return a generator which yields lines of a file

    Args:
        filename ([str]): the name of the file

    Yields:
        str: the line contents
    """
    with open(filename, 'r', newline='\n') as f:
        while (line := next(f)) != None:
            yield json.loads(line)
