import numpy as np

ZIGZAGINVERSE = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                   [2,  4,  7,  13, 16, 26, 29, 42],
                   [3,  8,  12, 17, 25, 30, 41, 43],
                   [9,  11, 18, 24, 31, 40, 44,53],
                   [10, 19, 23, 32, 39, 45, 52,54],
                   [20, 22, 33, 38, 46, 51, 55,60],
                   [21, 34, 37, 47, 50, 56, 59,61],
                   [35, 36, 48, 49, 57, 58, 62,63]])

ZIGZAGFLATINVERSE = ZIGZAGINVERSE.flatten()
ZIGZAGFLAT = np.argsort(ZIGZAGFLATINVERSE)

def zigzag_single(block):
    """
    ZigZag scan over a 8x8 2D array into a 64-element 1D array.
    Args:
        numpy.ndarray: 8x8 2D array
    Returns:
        numpy.ndarray: 64-element 1D array
    """
    return block.flatten()[ZIGZAGFLAT]

def inverse_zigzag_single(array):
    """
    Inverse ZigZag scan over 64-element 1D array into a 8x8 2D array.
    Args:
        numpy.ndarray: 64-element 1D array
    Returns:
        numpy.ndarray: 8x8 2D array
    """
    return array[ZIGZAGFLATINVERSE].reshape([8,8])

if __name__ == "__main__":
    test = np.random.randint(100, size=[8,8])
    print(np.array_equal(test, inverse_zigzag_single(zigzag_single(test))))