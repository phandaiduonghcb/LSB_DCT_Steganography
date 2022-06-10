from PIL import Image
import cv2
import numpy as np
from zigzag import *
from image_preparation import *
import tifffile
def get_message_bits(message):
    return ''.join([format(ord(x),'08b') for x in message])

def check_params(obj, image_mode, row, col, message_bits):
    assert 0 < obj.num_used_channels <4, f'num_used_channels must be <=3 and >=1!'
    if image_mode == 'L' and obj.num_used_channels != 1:
        raise Exception(f'num_used_channels must be 1 for L mode image!')
    elif isinstance(obj,DCT2):
        assert row*col*obj.num_used_channels/(obj.block_size**2) >= len(message_bits), 'Message is too long for the cover image!'
    else:
        raise Exception()

class DCT2:
    
    def __init__(self, quant_level=100, num_used_channels=1, n_bits_per_block=1, end_string='$t3g0'):
        '''
            block_size: Kích thước block để thực hiện dct
            quant_level: 10 hoặc 50 
            num_used_channels: số lượng giá trị trong 1 pixel được dùng để chứa 1 bit message
            end_string: thêm vào cuối message, dùng để biết điểm kết thúc của message khi trích xuất từ stego image.
        '''
        self.quant_level = quant_level
        self.num_used_channels = num_used_channels
        self.end_string=end_string
        self.block_size=8
        if n_bits_per_block > self.block_size**2 - 1:
            raise Exception(f'n_bits_per_block must be smaller than {self.block_size**2 - 1}')
        self.n_bits_per_block =n_bits_per_block
        self.quantization_table = self._get_quantization_matrix()
        self.T = self._get_DCT2_matrix()
    def __repr__(self) -> str:
        return f'DCT2(num_used_channels={self.num_used_channels}, quant_level={self.quant_level}, n_bits_per_block={self.n_bits_per_block})'
    def _get_DCT2_matrix(self):
        '''Trả về DCT matrix dựa trên block size'''
        m = [[1/np.sqrt(self.block_size)]*self.block_size]
        for i in range(1,self.block_size):
            row = []
            for j in range(self.block_size):
                value = np.sqrt(2/self.block_size) * np.cos(((2*j+1)*i*np.pi)/(2*self.block_size))
                row.append(value)
            m.append(row)
        return np.array(m)
    def _get_quantization_matrix(self):
        Q = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                      [12,  12,  14,  19,  26,  58,  60,  55],
                      [14,  13,  16,  24,  40,  57,  69,  56],
                      [14,  17,  22,  29,  51,  87,  80,  62],
                      [18,  22,  37,  56,  68,  109, 103, 77],
                      [24,  35,  55,  64,  81,  104, 113, 92],
                      [49,  64,  78,  87,  103, 121, 120, 101],
                      [72,  92,  95,  98,  112, 100, 103, 99]])

        if self.quant_level == 50:
            return Q
        elif self.quant_level == 100:
            return np.array([[1]*self.block_size]*self.block_size)
        elif self.quant_level > 50:
            Q = Q * np.divide(100 - self.quant_level,  50)
            Q = np.where(Q > 255, 255, Q)
            return Q
        else:
            Q = np.divide(Q * 50, self.quant_level)
            Q = np.where(Q > 255, 255, Q)
            return Q

    def _dc_transform(self, block):
        '''Perform DCT on a 2d block'''
        newBlock = self.T @ block @ self.T.transpose()
        return newBlock
    def _quantize(self, block):
        return np.round(block/self.quantization_table)
    def _inverse_quantize(self, block):
        return block * self.quantization_table
    def _inverse_dc_transform(self, block):
        return self.T.transpose() @ block @ self.T
    def _replace_DC_bit(self,block, bit):
        if (bit =='1' and block[0][0]%2==0):
            block[0][0] += 1
        elif (bit =='0' and block[0][0]%2==1):
            block[0][0] -=1
    def _replace_AC_bits(self,block, bits):
        array = zigzag_single(block)
        for i in range(len(bits)):
            if array[i+1]%2==0 and bits[i]=='1':
                array[i+1]+=1
            elif array[i+1]%2==1 and bits[i] =='0':
                array[i+1]-=1
        return inverse_zigzag_single(array)
    def _encode_2d_matrix(self,matrix, bits):
        '''Encode ma trận 2 chiều với message là dãy bits'''
        row,col = matrix.shape
        originalBlocks = divide_matrix_to_blocks(matrix, self.block_size, self.block_size)
        dctBlocks = [np.round(self._dc_transform(block-128)) for block in originalBlocks]
        quantizedBlocks = [self._quantize(block) for block in dctBlocks ]

        index = 0
        for i in range(1,len(quantizedBlocks)):
            nextIndex = min(index+self.n_bits_per_block, len(bits))
            quantizedBlocks[i] = self._replace_AC_bits(quantizedBlocks[i],bits[index:nextIndex])
            index=nextIndex
            if index == len(bits):
                break

        iquantizedBlocks = [self._inverse_quantize(block) for block in quantizedBlocks]
        reconstructedBlocks = [self._inverse_dc_transform(block)+128 for block in iquantizedBlocks]
        reconstructedBlocks[0] = originalBlocks[0]
        newChannel = merge_blocks_to_matrix(reconstructedBlocks, row, col)

        mx = np.amax(newChannel)
        mn = np.amin(newChannel) 
        newChannel = (newChannel - mn) / (mx - mn)

        newChannel[0][0] = mx/500
        newChannel[1][0] = mn/500
        return newChannel.astype(np.float32), index

    def encode(self, img_path, message, dest_path, grayscale=False):
        """
            img_path: PIL Image
            message: String
            dest_path: Ví trí lưu stego image (encoded image)
            grayscale: Convert cover image thành grayscale trước khi encode.
        """
        
        img = Image.open(img_path)
        if grayscale:
            img = convert_to_grayscale(img)

        mode = img.mode
        message += self.end_string
        bits = get_message_bits(message)
        img = resize(img, self.block_size)
        col, row = img.size

        check_params(self, mode, row, col, bits)
        
        bitIndex=0
        if mode == 'RGB':
            img = np.array(list(img.getdata())).reshape(row,col,3)
            channels = list(cv2.split(img))
            for i in range(self.num_used_channels):
                channels[i], bitIndex = self._encode_2d_matrix(channels[i], bits)
                bits=bits[bitIndex:]
                if not bits:
                    break
            for i in range(len(channels)):
                if channels[i].dtype == 'int32':
                    channels[i]=(channels[i]/255).astype(np.float32)
            newImg = np.dstack((channels[0],channels[1],channels[2])).astype(np.float32)
        elif mode == 'L':
            img = np.array(list(img.getdata())).reshape(row,col)
            newImg, _ = self._encode_2d_matrix(img,bits)
            newImg = newImg.astype(np.float32)
        else:
            raise Exception('Unsupported image mode!')

        tifffile.imwrite(dest_path, newImg)

    def decode(self,img_path):
        
        img = tifffile.imread(img_path)
        if len(img.shape)==3:
            mode = 'RGB'
        elif len(img.shape)==2:
            mode = 'L'
        row, col = img.shape[:2]
        if mode == 'RGB':
            channels = [img[:,:,0],img[:,:,1],img[:,:,2]]
            b = ''
            msg = ''
            for i in range(self.num_used_channels):
                matrix = channels[i]
                mx=500*matrix[0][0]
                mn=500*matrix[1][0]
                matrix = (matrix*(mx-mn)) + mn
                originalBlocks = divide_matrix_to_blocks(matrix, self.block_size, self.block_size)
                dctBlocks = [np.round(self._dc_transform(block-128)) for block in originalBlocks]
                quantizedBlocks = [self._quantize(dctBlock) for dctBlock in dctBlocks]
                
                for block in quantizedBlocks[1:]:
                    array = zigzag_single(block)
                    for i in range(1,self.n_bits_per_block+1):
                        b += str(int(array[i]%2))
                        if len(b)==8:
                            msg += chr(int(b,2))
                            b =''
                            if msg[-len(self.end_string):] == self.end_string:
                                return msg[:-len(self.end_string)]
            return None
        elif mode == 'L':
            matrix = img
            mx=500*matrix[0][0]
            mn=500*matrix[1][0]
            matrix = (matrix*(mx-mn)) + mn
            originalBlocks = divide_matrix_to_blocks(matrix, self.block_size, self.block_size)
            dctBlocks = [np.round(self._dc_transform(block-128)) for block in originalBlocks]
            quantizedBlocks = [self._quantize(dctBlock) for dctBlock in dctBlocks]
            
            b=''
            msg=''
            for block in quantizedBlocks[1:]:
                array = zigzag_single(block)
                for i in range(1,self.n_bits_per_block+1):
                    b += str(int(array[i]%2))
                    if len(b)==8:
                        msg += chr(int(b,2))
                        b =''
                        if msg[-len(self.end_string):] == self.end_string:
                            return msg[:-len(self.end_string)]
            return None
        else:
            raise Exception('Unsupported image mode!')

