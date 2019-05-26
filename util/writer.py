# -*- coding: utf-8 -*-

import os
from google.protobuf import text_format

def write_file(file_content, path):
    """Writes file content to the path and creates all intermediate directories.
    
    Raises:
        OSError: path does not specify a valid path to a file. 

    Args:   
        file_content: string, bythe string content of the file which will be written to 
            the path.
        path: string, path to the file to which the file content will be written. 
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory) # might throw OSError
    
    with open(path, "w") as output_file:
        output_file.write(file_content)
    
def write_proto_as_text(message, path):
    """Writes protocol buffer message to the path in pbtxt format. 
    
    Raises:
        OSError: path does not specify a path to a file.
    
    Args:
        message: a protocol buffer message.
        path: string, path to the file. 
    """
    with open(path, "w") as output_file:
        text_format.PrintMessage(message, output_file, as_utf8=True)

def write_proto_as_proto(message, path):
    """Serializes a protocol buffer message.
    
    Args:
        message: a protocol buffer message
        path: string, path to the output file
    """
    write_file(message.SerializeToString(), path)
    