1. The debug sub-directory under object_detection contains samples of the various output from running the Tensorflow object detector on an image or frame from a video

2. test_ images and test_videos are the sample input files used for testing the apps. The results are stored in the output folder nested in the object_detection directory

3. Stack overflow article on interpretting TensorFlow bounding box coordinates https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial. Here is another https://stackoverflow.com/questions/47110528/return-coordinates-for-bounding-boxes-googles-object-detection-api

4. Easy to follo tutorial on resizing images using OpenCV https://medium.com/@manivannan_data/resize-image-using-opencv-python-d2cdbbc480f0

5. The higher the resolution and larger the camera's field of vision, the higher the skip frame rate can be

6. Avg. FPS while processing 1920 x 1080 images is ~3. Reducing this to 490 X 360 increases this between 6-8 FPS. This can be further improved by running the solution with debug mode set to false as this reduces messages printed to the console as well as drawing boundary lines. 4 parallel processes were used for benchmarking

7. Skip frame rate, determining the counter line and resizing of images is something that will have to be calibrated (via test runs) for each production set up. The solution takes user input parameters to make necessary adjustments

8. While reducing the image dimensions improves FPS, it will adversely impact OCR since it requires high res images of the text region to identify the characters.

9. Crop and extract images - https://www.codementor.io/innat_2k14/extract-a-particular-object-from-images-using-opencv-in-python-jfogyig5u and https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

10. Example of implementing Factory Method Pattern in Python - https://realpython.com/factory-method-python/