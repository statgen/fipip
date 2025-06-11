import gzip

def flexopen(filename, mode='rt'):
    """
    Opens either a plain or gzip-compressed file, depending on
    whether filename ends with '.gz'.
    
    :param filename: The path to the file.
    :param mode: The file mode (e.g., 'r', 'rt', 'w', 'wt', etc.).
    :return: A file-like object (context manager).
    """
    if filename.endswith('.gz'):
        # You may choose to only add 'encoding' in text modes
        # if necessary, but below is a straightforward approach.
        return gzip.open(filename, mode, encoding='utf-8' if 't' in mode else None)
    else:
        return open(filename, mode, encoding='utf-8' if 't' in mode else None)
