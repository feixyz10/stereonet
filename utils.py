import sys
import numpy as np


def load_pfm(file_name):
    # ref: https://gist.github.com/chpatrick/8935738#file-python-pfm-numpy-library
    with open(file_name, 'rb') as fin:
        header = fin.readline().decode('utf-8').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        size = fin.readline().decode('utf-8').strip().split()
        W, H = int(size[0]), int(size[1])

        scale = float(fin.readline().decode('utf-8').strip())
        if scale < 0:  # little-endian
            end = '<'
            scale = -scale
        else:
            end = '>'  # big-endian

        img = np.fromfile(fin, end + 'f')
        shape = (H, W, 3) if color else (H, W)
        return np.reshape(img, shape) * scale


def save_pfm(file_name, img, scale=1.0):
    #ref: https://gist.github.com/chpatrick/8935738#file-python-pfm-numpy-library

    with open(file_name, 'wb') as fout:
        if img.dtype.name != 'float32':
            raise Exception('img dtype must be float32.')

        if len(img.shape) == 3 and img.shape[2] == 3:
            color = True
        elif len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 1:
            color = False
        else:
            raise Exception('img must have H x W x 3, H x W x 1 or H x W dimensions.')

        fout.write('PF\n' if color else 'Pf\n')
        fout.write('%d %d\n' % (img.shape[1], img.shape[0]))

        end = img.dtype.byteorder

        if end == '<' or end == '=' and sys.byteorder == 'little':
            scale = -scale

        fout.write('%f\n' % scale)

        img.tofile(fout)
