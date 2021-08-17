
import gc, os, sys, glob, argparse, utils
import subprocess
from config import root_direc
import cv2
from convert2digital import create_digital
from centering_correction import chess_mapping
import shutil


if __name__ == "__main__":

	p = argparse.ArgumentParser(description=\
	'Convert a physical chessboard image into a digital one')

	p.add_argument('--source', type=str, default='data/images', dest = 'image_source', help='source')
	p.add_argument('--destination', type=str, default='data/images', dest = 'image_dest', help='destination')

	args = p.parse_args(); src = str(args.image_source); dst = str(args.image_dest)

	shutil.rmtree(root_direc + r"\yolov5\runs\detect\exp")
	os.chdir("yolov5")

	os.system(r"python detect.py --weights " + root_direc + r"\models\yolov5s_model\weights\best.pt  --img 800 --conf 0.4 --save-txt --source " + src)

	os.chdir( root_direc)
	img = cv2.imread(src)
	squares, out = chess_mapping(img)
	board = create_digital(squares, out)

	board.save(dst + "\out.png")
