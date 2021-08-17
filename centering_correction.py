from neural_chessboard import pSLID, SLID, slid_tendency, LAPS, LLR, llr_pad, crop, slid_canny, root_direc
import cv2, numpy as np
import math
import os

def centreBoard(img, dp):

  height, width, _ = np.shape(img)
  scale_f = dp/width
  #img = cv2.resize(img, (int(scale_f * width), int(scale_f * height)), interpolation = cv2.INTER_LANCZOS4)

  segments = pSLID(img)
  raw_lines = SLID(img, segments)
  lines = slid_tendency(raw_lines)
  print(lines)

  points = LAPS(img, lines)

  inner_points = LLR(img, points, lines)
  four_points = llr_pad(inner_points, img) # padcrop

  delta = 0
  out, _ = crop(img, four_points, [delta,delta,delta,delta] )

  """
  corners = makeList(four_points)
  corners2 = makeList(inner_points)

  print(corners)

  out = warpImage(img, corners, dp)
  """



  #img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LANCZOS4)
  #cv2_imshow(out)
  return out, four_points

def crop_borders(img, adjust_f = 1):

    height, width, _ = np.shape(img)

    MORPH = 9
    CANNY = 84
    HOUGH = 25
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = slid_canny(thresh)
    #cv2_imshow(edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    #edges = cv2.dilate(edges, kernel)

    lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    for line in lines[0]:
         cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                         (255,0,0), 2, 8)
         
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    rects = []
    for cont in contours:
        cnt = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        if len(cnt) == 4 and cv2.contourArea(cnt) > (height*width/64 * adjust_f) and cv2.isContourConvex(cnt):
          rects.append(cnt)

    return rects[0]


def warp_point(M, pts):
  y = M.dot(np.array([pts[0], pts[1], 1]))
  return [ int(y[0]/y[2]) , int(y[1]/y[2]) ]

def square_area(img, fact = 0.8):

  rect = crop_borders(img, fact)
  #print(rect , height, width)

  y_arr = []
  x_arr = []

  for p in rect:
    y_arr.append(p[1])
    x_arr.append(p[0])

  points = [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]

  square_width = points[1] - points[0]
  square_height = points[3] - points[2]

  return square_width * square_height

def adjust_img(img, out, four_points):
  rect = crop_borders(out, 0.8)
  height, width, _ = np.shape(out)

  y_arr = []
  x_arr = []

  for p in rect:
    y_arr.append(p[1])
    x_arr.append(p[0])

  points = [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]

  delta = 20
  dx =  [delta,delta,delta,delta] 
  adjust_factor = 500
  adj_fact_2 = 1
  dx[0] += (points[1] - points[0] + width/adjust_factor) - points[0] % (points[1] - points[0] + width/adjust_factor)
  dx[1] += (points[1] - points[0] + width/adjust_factor) - (width - points[1]) % (points[1] - points[0] + width/adjust_factor)
  dx[2] += (points[3] - points[2] + height/adjust_factor * adj_fact_2) - points[2] % (points[3] - points[2] + height/adjust_factor)
  dx[3] += (points[3] - points[2] + height/adjust_factor) - (height - points[3]) % (points[3] - points[2] + height/adjust_factor)

  out, M = crop(img, four_points, dx)

  return out, M

def crop_excess(img):

  height, width, _ = np.shape(img)
  rect = crop_borders(img, 0.65)
  print(rect , height, width)

  y_arr = []
  x_arr = []

  for p in rect:
    y_arr.append(p[1])
    x_arr.append(p[0])

  points = [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]

  square_width = points[1] - points[0]
  square_height = points[3] - points[2]

  """
  adj = 0
  if( ( height ) > square_height*8):
    adj = height - 8*square_height

  board_top_x = int( ( (points[0]) % square_width )  )
  board_top_y = adj
  """

  delta = [int(( points[2] % square_height ) - 0.06 * square_height), 
           int(( points[2] % square_height ) + 8.04 * square_height),
           int( points[0] % square_width ),
           int(( points[0] % square_width ) + 8.05 * square_width)]

  return img[delta[0] : delta[1], delta[2] : delta[3]], delta


def split_to_squares(im, crop_delta, p_loc):

  height, width, _ = np.shape(im)

  squares = []

  M = width//8
  N = height//8

  for y in range(0, height, N):
    for x in range(0,width,M):

          tiles = [ int(x + M/2) + crop_delta[2] , int(y + N/2) + crop_delta[0]]

          squares.append(tiles)

  piece_square = [[0,0,0]]*64

  for locs in p_loc:
    min_dist = 500000
    min_x = 0
    for x in range(0,64):
      dist = math.sqrt((locs[1] - squares[x][0]) ** 2 + (locs[2] - squares[x][1]) ** 2 )
      if dist < min_dist:

          min_dist = dist
          min_x = x

    piece_square[min_x] = [locs[0], squares[min_x][0] - crop_delta[2] , squares[min_x][1] - crop_delta[0]]


  return piece_square

def piece_locations( M, img ):

  with open(root_direc + r"\yolov5\runs\detect\exp\labels\out.txt") as f:
      piece_loc = [[float(x) for x in line.split()] for line in f]

  #print(piece_loc)
  height, width, _ = np.shape(img)

  pieces_coord = []

  for piece in piece_loc:
    x_loc = int( ( piece[1] ) * width)
    y_loc = int( ( piece[2] + piece[4]/2 - 0.02 ) * height)

    point = warp_point(M, [x_loc, y_loc])

    """
    cv2.drawMarker(img, (point[0], point[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
      markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    """
    pieces_coord.append([ int(piece[0]), point[0], point[1] ])
    

  #cv2_imshow(img)

  return pieces_coord

def comp_area(src, centred_img, four_points):

  img, M = adjust_img(src, centred_img, four_points)
  p_loc = piece_locations(M, src)
  img_2 , crop_delta = crop_excess(img.copy())

  height, width, _ = np.shape(img)
  height_2, width_2, _ = np.shape(img_2)

  area = height*width
  area_2 = height_2 * width_2 

  diff_1 = abs(64*square_area(img, 0.65) - area)
  diff_2 = abs(64*square_area(img_2) - area_2)
  print(square_area(img, 0.65), square_area(img_2))

  if (diff_1 < diff_2):
     return img,[0,0,0,0]

  return img_2, crop_delta, p_loc

def chess_mapping(img):
  
  out, four_points = centreBoard( img , 1000)
  #cv2_imshow(out)
  out, crop_delta, p_loc = comp_area(img, out, four_points)
  #cv2_imshow(out)
  #cv2.imwrite("/content/drive/MyDrive/yolov5/runs/detect/cropped_boards/out.jpg", out)
  
  squares = split_to_squares(out, crop_delta, p_loc)

  for tiles in squares:
    #print(tiles)
    cv2.drawMarker(out, (tiles[1], tiles[2]),(0,0,255), markerType=cv2.MARKER_STAR, 
    markerSize=40, thickness=2, line_type=cv2.LINE_AA)
  cv2.imshow("image", out)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return squares, out
