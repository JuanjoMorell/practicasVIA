from puzzle import extract_digit
from puzzle import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2
from tesserocr import PyTessBaseAPI, RIL, iterate_level
import tesserocr
from PIL import Image

ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="imagen")
ap.add_argument("-d","--debug", type=int, default=1, help="debug")
args = vars(ap.parse_args())

print("[INFO] Loading digit classifier...")
model = load_model("output/digit23.h5")
#tesserocr.SetVariable("classify_bln_numeric_mode", "1")
#tesseract = tesserocr.PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_CHAR, oem=tesserocr.OEM.DEFAULT)

#print("[INFO] Processing image...")
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    image = imutils.resize(frame, width=600)

    (puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)
    

    board = np.zeros((9,9), dtype="int")

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    cellLocs = []

    for y in range(0,9):
        row = []

        for x in range(0,9):
            startX = x * stepX
            startY = y * stepY
            endX = (x+1) * stepX
            endY = (y+1) * stepY

            row.append((startX, startY, endX, endY))

            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=args["debug"] > 0)

            if digit is not None:
                roi = cv2.resize(digit, (28,28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                #cv2.imshow("a", digit)
                #cv2.waitKey(0)

                pred = model.predict(roi).argmax(axis=1)
                
                #tesseract.SetImage(Image.fromarray(digit))

                #pred = tesseract.GetUTF8Text()

                    # TODO Cambiar un modelo por un ocr
                print(pred)
                numero = pred
                board[y,x] = numero

        cellLocs.append(row)

    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3,3,board=board.tolist())
    puzzle.show()

    print("[INFO] Solving Sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        for (box, digit) in zip(cellRow, boardRow):
            startX, startY, endX, endY = box

            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            
            cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,255), 2)

    cv2.imshow("Sudoku Result", image)
    key = cv2.waitKey(0)
    if key == 27:
        break