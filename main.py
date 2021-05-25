import argparse
import draw_curve
from pathlib import Path


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('N', help='N = [4, 8, 16, 32...]', type=int)
  parser.add_argument('-i', help="image")

  args = parser.parse_args()
  
  if args.i:
      img = Path('.', args.i)
  else: 
      img = Path('.', 'img', 'lenna.png')
  
  draw_curve.draw(img, args.N)