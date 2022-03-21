# TODO: programatic myData parsing
# TODO: send roi data from regionselector script to txt file that this script reads
# TODO: pattern matching to recognize flipped, rotated forms etc
# TODO: PRE PROCESSING ALGORITH!!!!

import cv2
import numpy
from pytesseract import pytesseract
import re
import ast
import os

# Takes a cv2 image and pre-processes it in preparation for Tesseract OCR.
def pre_process_form(form):

    # 4-neighbors laplacian filter
    kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    form_sharp = cv2.filter2D(form, -1, kernel)
              
    # Convert the image to gray scale
    gray = cv2.cvtColor(form_sharp, cv2.COLOR_BGR2GRAY)

    form_width = form.shape[1]
    form_height = form.shape[0]
    # TODO: Breakpoint for if the image is already big enough, and computation to find scale value
    # to upscale the image to some consistent size.
    dim = (form_width * 2, form_height * 2)

    # Upscale the image to 5x its original size
    scaled_form = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    return scaled_form

name_regex = '([A-Z]+[,\.]+\s*[A-Z]+)'
npi_regex = '[0-9]{10}'

pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Match image to template
template = cv2.imread("hcfa_blank.png")
h, w, c = template.shape
print(w, h)
orb = cv2.ORB_create(100000)
key_point1, descriptor1 = orb.detectAndCompute(template, None)
#imKp1 = cv2.drawKeypoints(template, key_point1, None)


# Find all test images
path = "test images"
pictures = os.listdir(path)
print(pictures)

# Read roi from roi file
roi_file = open("roi.txt", "r")
roi_file_text = roi_file.read()
roi = ast.literal_eval(roi_file_text)
# roi = [[(16, 118), (100, 139), 'text', 'Name'], [(498, 528), (548, 542), 'text', 'NPI']]
print(roi)
roi_file.close()

myData = []

for j,y in enumerate(pictures):
    img = cv2.imread(path + "/" + y)
    key_point2, descriptor2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(descriptor2, descriptor1)
    matches = sorted(matches, key = lambda x: x.distance)
    # Take the top 25% keypoint matches.
    percentage = 25
    good = matches[:int(len(matches) * (percentage / 100))]
    imgMatch = cv2.drawMatches(img, key_point2, template, key_point1, good, None, flags = 2)

    src_points = numpy.float32([key_point2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = numpy.float32([key_point1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgShow = imgScan.copy()
    
    imgMask = numpy.zeros_like(imgShow)
    
    for x,r in enumerate(roi):
        # cv2.imshow(y, imgShow)
        # cv2.waitKey(0)
        
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0],r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # dim = (imgCrop.shape[1] * 2, imgCrop.shape[0] * 2)
        # imgCrop = cv2.resize(imgCrop, dim, interpolation = cv2.INTER_AREA)
        
        if r[2] == 'text' or r[2] == 'number':
            tesseractImage = pre_process_form(imgCrop)
            cv2.imshow(str(y + r[3]), tesseractImage)
            myData.append((pytesseract.image_to_string(tesseractImage)))
        
        
print(myData)

# name = re.findall(name_regex, myData[0])[0]
# npi = re.findall(npi_regex, myData[1])

# names = []

# if name != None:
#    # TODO: Tesseract has a tendancy to read ',' as '.'. This may be too permissive.
#    if ", " in name:
#        names = name.split(', ')
#    elif ". " in name:
#        names = name.split('. ')

# if len(names) == 2:
#    print('LAST: ' + names[0])
#    print('FIRST: ' + names[1])

# if npi:
#    print('NPI: ' + npi[0])

cv2.waitKey(0)

    
    
    
