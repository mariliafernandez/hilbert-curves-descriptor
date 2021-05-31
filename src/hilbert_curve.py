# algorithm reference: http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

from PIL import Image, ImageDraw
from pathlib import Path, PurePath
import numpy as np

def index2xy(index, N, im_size=None):

    if im_size == None or N > im_size:
        N = im_size
        
    # x, y positions in N=2
    positions = [
        [0,0],
        [0,1],
        [1,1],
        [1,0]
    ]
    
    # last 2 bits = position in N=2
    x, y = positions[ index&3 ]
    
    # next 2 bits = position in current N
    index = index >> 2
    
    n=4
    while n <= N:
        n2 = n//2
        
        h = index&3
        
        # Bottom left
        if h == 0:
            x, y = y, x
            
        # Upper left
        elif h == 1:
            x, y = x, y+n2
            
        # Upper right
        elif h == 2:
            x, y = x+n2, y+n2

        # Bottom right
        elif h == 3:
            x, y  = 2*n2-1-y, n2-1-x
            
        index = index >> 2
        n *= 2
        
    
    x, y = im_size//N*x, im_size//N*y
        
    return x, y


def hilbert_order(N, im_size=None):
    img_curve = []

    for i in range(0, N*N):
        pixel_coord = index2xy(i, N, im_size)
        img_curve.append(pixel_coord)
    
    return np.asarray(img_curve).flatten()


def draw_curve_on_img(img, N):
    curve = hilbert_order(N, img.size[0])
    draw = ImageDraw.Draw(img)

    draw.line(list(curve), width=1, fill=128)
    
    return img


def draw(img_path, N):
    label  = PurePath(img_path).parts[-2]
    name = img_path.stem

    with Image.open(img_path) as f:
        img = f.copy()

    img = draw_curve_on_img(img, N)
    img.save(Path(f'img/hilbert/cifar/{label}/{name}_{N}.png'))


    
