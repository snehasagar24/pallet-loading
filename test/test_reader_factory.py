from object_detection.ocr import readers
from object_detection.ocr.thumbnail import Thumbnails
import os
import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fName',"--folderName", type=str,
                       help="name of directory to store thumbnails", required=True)
    parser.add_argument("-o", "--output", type=str,
	                    help="path to store solution results")
    parser.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
    parser.add_argument("-ctxt", "--min-confidence", type=float, default=0.5,
	                    help="minimum probability required to inspect a region for text")
    parser.add_argument("-ocrW", "--ocrWidth", type=int, default=320,
	                    help="nearest multiple of 32 for resized width")
    parser.add_argument("-ocrH", "--ocrHeight", type=int, default=320,
	                    help="nearest multiple of 32 for resized height")
    parser.add_argument("-p", "--padding", type=float, default=0.0,
	                    help="amount of padding to add to each border of ROI")
    parser.add_argument("-ct", "--codeType", type=str,
                    default="barcode",help="options are ocr, barcode or qrcode")

    args = parser.parse_args()

    #create results folder structure
    result_root = os.path.join(args.output,args.folderName)

    log_path = os.path.join(result_root, "Logs")
    #check if the directory exists
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = os.path.join(log_path,(args.folderName + ".log"))

    #this block of code will read CBU code, barcode or QR code depending on the input parameter
    start = time.time()

    thumbnailDir = os.path.join(result_root, "Thumbnails")
    #check if the directory exists
    if not os.path.exists(thumbnailDir):
        os.makedirs(thumbnailDir)

    ocrSaveDir = os.path.join(result_root, "Product_Code_Detection")
    if not os.path.exists(ocrSaveDir):
        os.mkdir(ocrSaveDir)

    #retrieve list of image files for which we will read the product codes
    image_files_list = [f for f in os.listdir(thumbnailDir) if os.path.isfile(os.path.join(thumbnailDir, f))]

    print("Scanning product codes...")

    thumbnails = Thumbnails(imgSourcePath=thumbnailDir, thumbnailList=image_files_list, saveDir=ocrSaveDir, resize_height=args.ocrHeight, resize_width=args.ocrWidth, tf_model_path=args.east, ocr_confidence=args.min_confidence, padding=args.padding)

    pCode_reader = readers.ObjectReader()

    pCode_reader.read(readable=thumbnails, format=args.codeType)

    with open(log_file_name, 'a+') as f:
        print(f'[INFO] product code type: {args.codeType}', file=f)
        print(f'[INFO] total time taken for reading product codes: {time.time() - start: .3f} seconds', file=f)

    print("Scanning completed!")