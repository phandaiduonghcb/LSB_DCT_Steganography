import cv2
from PIL import Image, ImageOps
import numpy as np

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def resize(image, block_size):
    col, row = image.size
    if row%block_size != 0:
      row = row + (block_size - row%block_size)
    if col%block_size != 0:
      col = col + (block_size - col%block_size)
    return image.resize((col,row))

def divide_matrix_to_blocks(matrix, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = matrix.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (matrix.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))

def merge_blocks_to_matrix(blocks, row, col):
    def chunks(l, n):
      m = int(n)
      for i in range(0, len(l), m):
          yield l[i:i + m]
    
    block_size = blocks[0].shape[0]
    newMatrix = []
    for chunkRowBlocks in chunks(blocks, col/block_size):
        for rowBlockNum in range(block_size):
            for block in chunkRowBlocks:
                newMatrix.extend(block[rowBlockNum])
    newMatrix = np.array(newMatrix).reshape(row, col)
    return newMatrix

# def get_YCrCb_array(path):
#   im = cv2.imread(path)
#   im = cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
#   return im

# def convert_YCrCb_to_RGB(image):
#   return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
