import os
import cv2
import numpy as np
from object_detection.trackAndDetect import trackingPoolWorker as tpw


def getThumbnail(image,bbox):
    '''Extract object thumbnails from an image using the bouding box coordinates applied by an object detector. Save the thumbnail to a specified location on disk

    Parameters:

        image: array<float> - image/video frame

        bbox: array<int>  - coordinates of the 4 end points of the rectangle that bounds a detected object

    Returns:

        cropped_image: array<float> - thumbnail of the object detected extracted from the source image/video frame

    NOTE:
                ymin = top =    box[0]

                xmin = left =   box[1]

                ymax = bottom = box[2]

                xmax = right =  box[3]

                TensorFlow requires bounding boxes in the format [ymin, xmin, ymax, xmax]
                dlib.rectangle requires boxes in the format [left,top,right,bottom]
                
                '''

    
    #format the coordinates as expected by openCV
    cropped_image = image[bbox[0]:bbox[2],bbox[1]:bbox[3], :]

    return cropped_image

def thumbnailIterator(img, trackableObj, image_height, image_width, bbox_index=None):
    '''A wrapper of the getThumbnail method that iterates over a list of rectangle coordinates and returns a list of extracted thumbnails

    Parameters:

        img: Numpy representation of an image from which we want to extract thumbnails of detected objects

        bbox_index: A list of integer indexes which correspond to high confidence detection scores

        trackableObj: A custom class that holds output from applyign the object detector on an image/video frame

        image_heigt: An integer value representing the original height of the image passed for processing

        image_width: An integer value representing the original width of the image passed for processing

    Returns:

        thumbnail_list: A list of normalized thumbnails extracted from the input image
        '''

    thumbnail_list = []

    #extract thumbnail
    if bbox_index:
        for idx in bbox_index:
            each_box = trackableObj._boxes[0][idx]
            normalized_bbox = tpw.normalizeBBoxCoordinates(each_box,image_height,image_width,debug=False)
            thumbnail_list.append(getThumbnail(img, normalized_bbox))
    else:
        for each_box in trackableObj._boxes[0]:
            normalized_bbox = tpw.normalizeBBoxCoordinates(each_box,image_height,image_width,debug=False)
            thumbnail_list.append(getThumbnail(img, normalized_bbox))

    return thumbnail_list

def writeThumbnailToDisk(thumbnailList, fileName, frameNum, imgNum, saveDir):
    '''This method writes extracted thumbnails to a specified location on disk

    Parameters:

        thumbnailList: A list containing the numpy array representation of the thumbnails that need to be written to disk

        fileName: A string representing the name of the source file which is being processed

        frameNum: An integer value for the video frame that is being processed

        imgNum: A multiprocessing Value object that is used to assign unique integer values to the thumbnails written to disk

        saveDir: A string representing the location on disk where the thumbnail is to be written
        '''

    #print(f'Number of thumbnails: {len(thumbnailList)}')

    outPath = os.path.join(saveDir,'Thumbnails')

    #check if the directory exists
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for each_thumbnail in thumbnailList:
        #create output file path
        outFileName = f'{fileName}_Frame-{str(frameNum)}_Image-{str(imgNum.value)}.jpg'
        output_thumbnail = cv2.cvtColor(each_thumbnail, cv2.COLOR_RGB2BGR)
        outFilePath = os.path.join(outPath,outFileName)
        imgNum.update()

        cv2.imwrite(outFilePath, output_thumbnail)