"""
HW-1 list functions. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

import random


def list_stats(values):
    """Returns the min, max, average, and sum of the values in the given
    list as a tuple (min, max, avg, sum).
      
    Args:
        values: The list of values to compute statistics over.

    Notes:
        Tuple (None, None, None, None) returned if values is empty.
        Assumes a list of numerical values. 

    Example: 
        >>> list_stats([1, 2, 3])
        (1, 3, 2.0, 6)

    """
    # Try list to make sure values are valid for stat computations
    try:
        # Collect stats and store in vars
        val_min = min(values)
        val_max = max(values)
        val_avg = sum(values) / len(values)
        val_sum = sum(values)

        # Return stats tuple
        return (val_min, val_max, val_avg, val_sum)
    
    except:
        # Invalid or empty tuple returns None tuple
        return (None, None, None, None)


def convert_numeric(value):
    """Returns corresponding numeric value for given string value.

    Args:
        value: The string value to convert.

    Notes:
        Given value returned if  cannot be converted to int or float.

    Examples:
        >>> convert_numeric('abc')
        'abc'
        >>> convert_numeric('42')
        42
        >>> convert_numeric('3.14')
        3.14

    """
    # Try if string has integer value
    try:
        int_val = int(value)
        return int_val

    except:
        # If integer fails, try float value
        try:
            float_val = float(value)
            return float_val
        
        except:
            # Return value that cannot be converted to int or float
            return value


def random_matrix_for(m, n):
    """Return an m x n matrix as a list of lists containing randomly
    generated integer values.

    Args:
        m: The number of rows. 
        n: The number of columns.

    Notes:
        Values are from 0 up to but not including m*n.
    
    Example:
        >>> random_matrix_for(2, 3)
        [[2, 1, 0], [3, 6, 4]]

    """
    # Initialize random matrix
    random_matrix = []

    # Append random integer rows with for loops
    for i in range(m):
        row = []

        for j in range(n):
            row.append(random.randint(0, m*n - 1))

        random_matrix.append(row)
    
    # Return filled random matrix
    return random_matrix



def random_matrix_comp(m, n):
    """Return an m x n matrix as a list of lists containing randomly
    generated integer values.

    Args:
        m: The number of rows. 
        n: The number of columns.

    Notes:
        Values are from 0 up to but not including m*n.
    
    Example:
        >>> random_matrix_for(2, 3)
        [[2, 1, 0], [3, 6, 4]]

    """
    # Nested list comprehension used to fill each value in matrix with random integer
    return [[random.randint(0, m*n - 1) for i in range(n)] for j in range(m)]





def transpose_matrix(list_matrix): 
    """Return the transpose of the given matrix represented as a list of
    lists.

    Args:
        list_matrix: The list version of the matrix to transpose.

    Example: 
        >>> transpose_matrix([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]

    """
    # Transpose Matrix Dimensions
    trans_m = len(list_matrix[0])
    trans_n = len(list_matrix)

    # Load Transpose Dimension Matrix with -1
    trans_matrix = [[-1 for n in range(trans_n)] for m in range(trans_m)]
    
    # Change -1 values in initial trans_matrix with values given from list_matrix
    for i in range(len(list_matrix)):
        for j in range(len(list_matrix[0])):
            trans_matrix[j][i] = list_matrix[i][j]

    # Return transposed matrix
    return trans_matrix
        



def reshape_matrix(list_matrix, m, n):
    """Return a new matrix based on the given matrix but scaled to m rows
    and n columns.

    Args:
        list_matrix: The matrix to reshape.
        m: The new number of rows.
        n: The new number of columns.

    Notes:
        New rows or columns are filled with 0 values.

    Example: 
        >>> reshape_matrix([[1, 2, 3], [4, 5, 6]], 3, 2)
        [[1, 2], [4, 5], [0, 0]]

    """
    # Current list_matrix dimensions
    curr_m = len(list_matrix)
    curr_n = len(list_matrix[0])

    # Initialize reshape matrix with zeros
    new_matrix = [[0 for k in range(n)] for h in range(m)]

    # Change values in new_matrix based on values from list_matrix
    for i in range(m):       
        for j in range(n):
            # If list_matrix is not same size as given dimensions, fill extra spots with zero
            if i >= curr_m or j >= curr_n:
                new_matrix[i][j] = 0
            else:
                new_matrix[i][j] = list_matrix[i][j]
    
    # Return filled new_matrix
    return new_matrix
