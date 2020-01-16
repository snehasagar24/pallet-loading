from collections import OrderedDict, Counter
from pyzbar import pyzbar
from object_detection.ocr.text_detection_and_ocr import decode_predictions
from imutils.object_detection import non_max_suppression
import os
import json
import numpy as np
import pytesseract
import cv2

class ReaderFactory():
    """This class implements the reader factory"""
    
    def __init__(self):
        self._readers = {}

    
    def register_format(self, format, reader):
        self._readers[format] = reader

    def get_reader(self, format):
        reader = self._readers.get(format)
        if not reader:
            raise ValueError(format)
        return reader()

class OcrReader():
    '''This class performs Optical character recognition on text detected on an object'''

    def __init__(self):
        self._properties_dict = OrderedDict()
        self._results_json = None

    def start_object(self, imgSourcePath, thumbnailList, saveDir):
        '''This method initializes the propoerties dictionary with the following keys:
        
            inputPath: str - path where the thumbnails are located on disk

            thumbnails: list<str> - names of the thumbnail files which need to be scanned
            
            outputPath: str - location on disk where the JSON objects needs to written'''

        self._properties_dict["inputPath"] = imgSourcePath
        self._properties_dict["thumbnails"] = thumbnailList
        self._properties_dict["outputPath"] = saveDir

    def add_property(self, name, value):
        '''This method is a helper function to add more key-value pairs to the properties dictionary'''

        self._properties_dict[name] = value

    def _create_json(self):
        '''This method will perform text location and reading and prepare the JSON output
        
        Returns:
            
            result_json: dict<str,T> - JSON object containing the results of text location and reading'''

        #initialize dictionary to store results
        result_json = OrderedDict()

        #store frequency of each text detected, similar to word count
        result_json["code_frequency"] = []

        #nested ordered dictionary to store the names of files where text was located and OCR performed
        result_json["code_detected"] = OrderedDict()
        result_json["code_detected"]["total"] = 0
        result_json["code_detected"]["file_names"] = []

        #nested ordered dictionary to store the names of files where no text was located
        result_json["no_code_detected"] = OrderedDict()
        result_json["no_code_detected"]["total"] = 0
        result_json["no_code_detected"]["file_names"] = []

        #list to store dictionaries of bounding box coordinates and text detected
        result_json["results"] = []

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        #print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(self._properties_dict["tf_model_path"])

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")

        for each_file in  self._properties_dict["thumbnails"]:

            full_image_input_path = os.path.join(self._properties_dict["inputPath"], each_file)

            # load the input image and grab the image dimensions
            image = cv2.imread(full_image_input_path)
            orig = image.copy()
            (origH, origW) = image.shape[:2]

            # set the new width and height and then determine the ratio in change
            # for both the width and height
            (newW, newH) = (self._properties_dict["new_image_width"], self._properties_dict["new_image_height"])
            rW = origW / float(newW)
            rH = origH / float(newH)

            # resize the image and grab the new image dimensions
            image = cv2.resize(image, (newW, newH))
            (H, W) = image.shape[:2]

            # construct a blob from the image and then perform a forward pass of
            # the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	            (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry, self._properties_dict["ocr_confidence"])
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            # initialize the list of results
            results = []

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
	            # scale the bounding box coordinates based on the respective
	            # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

	            # in order to obtain a better OCR of the text we can potentially
	            # apply a bit of padding surrounding the bounding box -- here we
	            # are computing the deltas in both the x and y directions
                dX = int((endX - startX) * self._properties_dict["padding"])
                dY = int((endY - startY) * self._properties_dict["padding"])

	            # apply padding to each side of the bounding box, respectively
                startX = max(0, startX - dX)
                startY = max(0, startY - dY)
                endX = min(origW, endX + (dX * 2))
                endY = min(origH, endY + (dY * 2))

	            # extract the actual padded ROI
                roi = orig[startY:endY, startX:endX]

	            # extract text using tesseract
                text = pytesseract.image_to_string(roi, config=config)

	            # add the bounding box coordinates and OCR'd text to the list
	            # of results
                results.append(((startX, startY, endX, endY), text))

            #check if there are any results before proceeding
            if results:
                # sort the results bounding box coordinates from top to bottom
                results = sorted(results, key=lambda r:r[0][1])

                #create a copy of the input image
                output = orig.copy()

                #create an ordered dictionary for each set of file name, coordinates and text detected in an image 
                each_file_details = OrderedDict() 
                each_file_details["file_name"] = each_file
                each_file_details["text_detected"] = OrderedDict()

                # loop over the results
                for ((startX, startY, endX, endY), text) in results:
                # display the text OCR'd by Tesseract
                #print("OCR TEXT")
                #print("========")
                #print("{}\n".format(text))

                    # strip out non-ASCII text so we can draw the text on the image
                    # using OpenCV, then draw the text and a bounding box surrounding
                    # the text region of the input image
                    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                    cv2.rectangle(output, (startX, startY), (endX, endY),
	                    (0, 0, 255), 2)
                    cv2.putText(output, text, (startX, startY + 100),
	                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    #add coordinates and text from OCR
                    each_file_details["text_detected"]["startX"] = startX
                    each_file_details["text_detected"]["endX"] = endX
                    each_file_details["text_detected"]["startY"] = startY
                    each_file_details["text_detected"]["endY"] = endY
                    each_file_details["text_detected"]["text"] = text

                    #append the dictionary to the results list belonging to the parent dictionary
                    result_json["results"].append(each_file_details)

                    #add text to text frequency list
                    result_json["code_frequency"].append(text)

                    # show the output image - only use for debugging, comment out while using in production
                    #cv2.imshow("Text Detection", output)
                    #cv2.waitKey(0)

                result_json["code_detected"]["total"] += 1
                result_json["code_detected"]["file_names"].append(each_file)

                # write results to disk
                #full_image_output_path = os.path.join(args.outputPath, each_file)
                #cv2.imwrite(full_image_output_path, output)

            else:
                result_json["no_code_detected"]["total"] += 1
                result_json["no_code_detected"]["file_names"].append(each_file)

        cv2.destroyAllWindows()

        #convert text frequency list into a Counter and then a dictionary, to get key-value pairs of the type word:count
        result_json["code_frequency"] = dict(Counter(result_json["code_frequency"]))

        return result_json

    def to_json(self):
        '''Writes the JSON object containing OCR results to disk'''

        self._result_json = self._create_json()

        with open(os.path.join(self._properties_dict["outputPath"],"ocr_results.json"), 'w') as fout:
            json.dump(self._result_json, fout)

class BarcodeReader(OcrReader):
    '''This class inherits from OcrReader and overrides the _create_json and to_json methods'''
    
    def _create_json(self):
        '''Applies barcode or QR Code reading instead of OCR and writes the results to JSON
        
        Returns:
        
            result_json: dict<str,T> - JSON object containing the results of barcode location and reading
            
            '''

        #initialize dictionary to store results
        result_json = OrderedDict()

        #store frequency of each text detected, similar to word count
        result_json["code_frequency"] = []

        #nested ordered dictionary to store the names of files where text was located and OCR performed
        result_json["code_detected"] = OrderedDict()
        result_json["code_detected"]["total"] = 0
        result_json["code_detected"]["file_names"] = []

        #nested ordered dictionary to store the names of files where no text was located
        result_json["no_code_detected"] = OrderedDict()
        result_json["no_code_detected"]["total"] = 0
        result_json["no_code_detected"]["file_names"] = []

        #list to store dictionaries of bounding box coordinates and text detected
        result_json["results"] = []

        for each_file in  self._properties_dict["thumbnails"]:

            full_image_input_path = os.path.join(self._properties_dict["inputPath"], each_file)

            # load the input image
            image = cv2.imread(full_image_input_path)
 
            # find the barcodes in the image and decode each of the barcodes
            barcodes = pyzbar.decode(image)

            if barcodes:

                #create an ordered dictionary for each set of file name, coordinates and text detected in an image 
                each_file_details = OrderedDict() 
                each_file_details["file_name"] = each_file
                each_file_details["text_detected"] = OrderedDict()

                # loop over the detected barcodes
                for barcode in barcodes:
                    # extract the bounding box location of the barcode and draw the
                    # bounding box surrounding the barcode on the image
                    (x, y, w, h) = barcode.rect
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
                    # the barcode data is a bytes object so if we want to draw it on
                    # our output image we need to convert it to a string first
                    barcodeData = barcode.data.decode("utf-8")
                    barcodeType = barcode.type
 
                    # draw the barcode data and barcode type on the image
                    text = "{} ({})".format(barcodeData, barcodeType)
                    #cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    # 0.5, (0, 0, 255), 2)
 
                    # print the barcode type and data to the terminal
                    #print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

                    #add coordinates and text from OCR
                    each_file_details["text_detected"]["startX"] = x
                    each_file_details["text_detected"]["endX"] = x + w
                    each_file_details["text_detected"]["startY"] = y
                    each_file_details["text_detected"]["endY"] = y + h
                    each_file_details["text_detected"]["text"] = text

                    #append the dictionary to the results list belonging to the parent dictionary
                    result_json["results"].append(each_file_details)

                    #add text to text frequency list
                    result_json["code_frequency"].append(text)

                result_json["code_detected"]["total"] += 1
                result_json["code_detected"]["file_names"].append(each_file)

                # show the output image
                #cv2.imshow("Image", image)
                #cv2.waitKey(0)

            else:
                result_json["no_code_detected"]["total"] += 1
                result_json["no_code_detected"]["file_names"].append(each_file)

        #convert text frequency list into a Counter and then a dictionary, to get key-value pairs of the type word:count
        result_json["code_frequency"] = dict(Counter(result_json["code_frequency"]))

        return result_json

    def to_json(self):
        '''Writes the JSON object containing barcode results to disk'''

        self._result_json = self._create_json()

        with open(os.path.join(self._properties_dict["outputPath"],"barcode_results.json"), 'w') as fout:
            json.dump(self._result_json, fout)

class QrcReader(BarcodeReader):
    '''This class inherits from BarcodeReader and overrides the to_json method'''

    def to_json(self):
        '''Writes the JSON object containing QR code results to disk'''

        self._result_json = self._create_json()

        with open(os.path.join(self._properties_dict["outputPath"],"qr_code_results.json"), 'w') as fout:
            json.dump(self._result_json, fout)

factory = ReaderFactory()
factory.register_format('ocr', OcrReader)
factory.register_format('barcode', BarcodeReader)
factory.register_format('qrcode', QrcReader)

class ObjectReader():
    '''Factory object creator'''

    def read(self, readable, format):
        '''This method implements the execution of the factory method
        
        Parameters:
        
            readable: Thumbnails - this is the object on which we want to perform text location and reading
            
            format: str - the reader method we want to apply i.e. ocr, barcode or qrcode
            '''

        reader = factory.get_reader(format)
        readable.readCode(reader)
        reader.to_json()