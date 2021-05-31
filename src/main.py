import argparse
import hilbert_curve
from pathlib import Path


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('mode', type=int)
  parser.add_argument('N', help='N = [4, 8, 16, 32...]', type=int)
  parser.add_argument('-i', help="image")

  args = parser.parse_args()

  if args.i:
      img = Path(args.i)
  else: 
      img = Path('../img/lenna.png')
  
  if args.mode == 1:
      hilbert_curve.draw(img, args.N)

  
  elif args.mode == 2:
      hilbert_curve.hilbert_order(args.N, args.N)