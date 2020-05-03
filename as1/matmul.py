
import numpy as np
import matplotlib.pyplot as plt

'''
How might we solve a set of linear equations?
This is the simplest case: two relations and two
unknowns.

    3x0 + x1  = 9
    x0  + 2x1 = 8

Or in another form, we are solving for x where:

    Ax = b

        | 3 1 |
    A = | 1 2 |

        | 9 |
    b = | 8 |

'''

if __name__ == '__main__':

    A = np.array([
        [3, 1],
        [1, 2],
    ])

    b = np.array([9, 8])

    x = np.linalg.solve(A, b)

    # Note that @ is the matrix multiplication operator
    print(f'Solution: {x}\n\n'
        'Ax - b = \n\n'
       f'{A} {x} - {b} = \n\n'
       f'{A @ x} - {b} = \n\n'
       f'{A @ x - b}\n')

    '''
    We just directly solved this set of linear equations
    because we were fortunate enough to have the same number
    of equations as we had unknowns (aka our matrix was square).
    If we are not so fortunate however, we will have to use an
    approximation, like so.

    Let us solve the following set of equations:

    x0  + 2x1 + x2  = 4
    x0  + x1  + 2x2 = 3
    2x0 + x1  + x2  = 5
    x0  + x1  + x2  = 4

    Or alternately,

    Ax = b where

        |1 2 1|
        |1 1 2|
        |2 1 1|
    A = |1 1 1|

            | 4 |
            | 3 |
            | 5 |
    and b = | 4 |
    '''

    A = np.array([
        [1, 2, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 1, 1],
    ])

    b = np.array([
        4,
        3,
        5,
        4,
    ])

    '''
    This is the 'magic' function that will do all our dirty
    work for us.
    
    We don't expect the result of Ax - b to equal zero exactly,
    as this is a least-squares approximation at the solution.
    '''

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f'Solution: {x}\n\n'
        'Ax - b = \n\n'
       f'{A} {x} - {b} = \n\n'
       f'{A @ x} - {b} = \n\n'
       f'{A @ x - b}')
