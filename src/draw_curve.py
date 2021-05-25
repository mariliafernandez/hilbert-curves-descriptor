# algorithm reference: http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

from PIL import Image, ImageDraw
from pathlib import Path


def index2xy(index, N):
        
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
        
    
    x, y = 512//N*x, 512//N*y
        
    return x, y


def draw(img_path, N):
    prev_pixel = (0,0)

    with Image.open(img_path) as img:

        for i in range(0, N*N):
            curr_pixel = index2xy(i, N)
            
            draw = ImageDraw.Draw(img)
            draw.line(prev_pixel + curr_pixel, width=3, fill=128)
            
            prev_pixel = curr_pixel

        
        img.save( Path('.', 'img', 'hilbert_curves_N'+str(N)+'.png') ) 




    
