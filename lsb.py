from PIL import Image, ImageOps
import numpy as np
import cv2


def get_message_bits(message):
    return ''.join([format(ord(x),'08b') for x in message])

def check_params(obj, image_mode, row, col, message_bits):
    assert 0 < obj.num_used_channels <4, f'num_used_channels must be <=3 and >=1!'
    if image_mode == 'L' and obj.num_used_channels != 1:
        raise Exception(f'num_used_channels must be 1 for L mode image!')
    elif isinstance(obj,LSB):
        assert row*col*obj.num_used_channels >= len(message_bits), 'Message is too long for the cover image!'
    else:
        raise Exception()

class LSB():
  def __init__(self, num_used_channels=1, end_string='$t3g0') -> None:
    '''
      num_used_channels: số lượng giá trị trong 1 pixel được dùng để chứa 1 bit message
      end_string: thêm vào cuối message, dùng để biết điểm kết thúc của message khi trích xuất từ stego image.
    '''
    self.num_used_channels = num_used_channels
    self.end_string=end_string
  def __repr__(self) -> str:
     return f'LSB(num_used_channels={self.num_used_channels})'
  def encode(self, img_path, message, dest, grayscale=False):
    """
      img_path: Stego image path
      message: String
      dest: Ví trí lưu stego image (encoded image)
      grayscale: Convert cover image thành grayscale trước khi encode.
    """

    img = Image.open(img_path)
    if grayscale:
      img = ImageOps.grayscale(img)
    width, height = img.size
    array = np.array(list(img.getdata()))
    num_pixels = width*height

    message += self.end_string # End of the message
    b_message = ''.join([format(ord(x),'08b') for x in message]) # Convert message to bits
    check_params(self,img.mode, height, width, b_message)
    if img.mode == 'RGB':
      idx = 0
      # Loop through each pixel and change R, G, B lsb value based on self.num_used_channels
      for p in range(num_pixels):
        for i in range(self.num_used_channels):
          if idx < len(b_message):
            array[p][i] = int(bin(array[p][i])[2:-1] + b_message[idx], 2) # Replace LSB
            idx +=1
        if idx >= len(b_message):
          break
      array=array.reshape(height, width, 3)

    elif img.mode == 'L':

      idx = 0
      # Loop through each pixel and change its lsb value
      for p in range(num_pixels):
        if idx < len(b_message):
          array[p] = int(bin(array[p])[2:-1] + b_message[idx], 2)
          idx +=1
        if idx >= len(b_message):
          break
      array=array.reshape(height, width)
    else:
      raise Exception('Unsupported image mode')

    enc_img = Image.fromarray(array.astype('uint8'), img.mode)
    enc_img.save(dest)
    return enc_img

  def decode(self,img_path):
    '''Decode PIL image'''

    img = Image.open(img_path)
    width, height = img.size
    array = np.array(list(img.getdata()))
    num_pixels = width*height
    hidden_bits = ''
    if img.mode == 'RGB':
      for p in range(num_pixels):
        for i in range(self.num_used_channels):
          hidden_bits += bin(array[p][i])[-1]
      
    elif img.mode == 'L':
      for p in range(num_pixels):
        hidden_bits += bin(array[p])[-1]
    else:
      raise Exception('Unsupported image mode')
    
    hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]
    message = ""
    for i in range(len(hidden_bits)):
        if message[-len(self.end_string):] == self.end_string:
            break
        else:
            message += chr(int(hidden_bits[i], 2))
    if self.end_string in message:
      return message[:-len(self.end_string)]
    else:
      return None