
from PIL import Image
import neural_chessboard 
import cv2, numpy as np
from centering_correction import chess_mapping

root_direc = neural_chessboard.root_direc

piece_codes_num = [ r"\blank.png", r"\bB.png",
 r"\bK.png" , r"\bN.png" , r"\bP.png" , r"\bQ.png" ,r"\bR.png" , r"\wB.png" ,
 r"\wK.png" , r"\wN.png" ,r"\wP.png" , r"\wQ.png" , r"\wR.png" ]


def create_digital(squares, out):

    height, width, _ = np.shape(out)
    M = width//8
    N = height//8
    empty_board = Image.open( root_direc + r"\chessboard.png")
    empty_board = empty_board.convert("RGBA")
    #blank_board = Image.new("RGBA", (480, 480), (255, 255, 255))
    for i in range (0,64):
        square = squares[i]

        icon_loc = root_direc + r"\piece_icons" + piece_codes_num[square[0]]

        curr_square = Image.open(icon_loc)

        if square[0] != 0:
            empty_board.paste(curr_square, ( square[1]//M * 60 , square[2]//N  * 60 ), curr_square  )

    empty_board.show()

img = cv2.imread(root_direc + r"\testing_images\14.jpg")
squares, out = chess_mapping(img)
create_digital(squares, out)

