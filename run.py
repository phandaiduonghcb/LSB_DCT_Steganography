from lsb import LSB
from dct import DCT
from dct2 import DCT2
import matplotlib.pyplot as plt
import numpy as np
import glob

import random, string

def get_random_string(l):
    return ''.join([random.choice(string.printable) for i in range(l)])

paths = glob.glob('original_images/*')
names = ['lsb.png','dct.tiff','dct2.tiff','dct2_10.tiff','quantized_dct2.tiff']
steg_objs = [LSB(),DCT(),DCT2(),DCT2(n_bits_per_block=20),DCT2(quant_level=50)]

for path in paths:
    im = plt.imread(path)*255
    im = im.astype(np.uint8)
    print('### ' + path)
    print('SHAPE:',im.shape)

    # message = get_random_string(500)
    message = 'zaPiCQohmLYgLdaLqdZidOyMcUEzbmbRGlfBpQVEloXPkbCWcWMGfxfBufziabXJyPAdcVNlrLsILOPFVnLAbLNXUAPmNhBjSSMZscJoMqESuRJlpFyFEiyCEddkEWiQfOSKbLTTHEcRLKbuZsWOBBqruKumNTnHPolrrRUvTwNLzfvgzsPNbzbXQgQCfvnWLDEOmFwg'
    for i,steg_obj in enumerate(steg_objs):
        d = 'encoded_images/'+ path.split('/')[-1].split('.')[0] + '_' + names[i]
        steg_obj.encode(path, message,d)
        returnMessage = steg_obj.decode(d)

        assert message == returnMessage, f'{str(steg_obj)}: Messages dont match!'