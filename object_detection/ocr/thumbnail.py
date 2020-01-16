class Thumbnails():
    """This is the Factory Client Object that will call an instance of a text reader"""

    def __init__(self, imgSourcePath, thumbnailList, saveDir, resize_height=None, resize_width=None, tf_model_path=None, ocr_confidence=None, padding=None):
        '''The last 5 parameters resize_height, resize_width, tf_model_path, ocr_confidence and padding are required for OCR only. Barcode or QRC readers do not require these
        '''
        self._thumbnailList = thumbnailList
        self._saveDir = saveDir
        self._imgSourcePath = imgSourcePath
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._tf_model_path = tf_model_path
        self._ocr_confidence = ocr_confidence
        self._padding = padding

    def readCode(self, reader):
        '''This method initiates the properties dictionary of a particular reader format and inserts addtional properties required relevant to the reader format selected
        
        Parameters:
        
            reader: ReaderFactory() instance - valid options are OCR, Barcode and QRCode
            '''

        reader.start_object(self._imgSourcePath, self._thumbnailList, self._saveDir)
        reader.add_property("new_image_height", self._resize_height)
        reader.add_property("new_image_width", self._resize_width)
        reader.add_property("tf_model_path", self._tf_model_path)
        reader.add_property("ocr_confidence", self._ocr_confidence)
        reader.add_property("padding", self._padding)