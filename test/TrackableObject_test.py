from object_detection.trackAndDetect.trackingPoolWorker import TrackableObject
import numpy as np

boxes = np.array([[[0.3745459 , 0.33260322, 0.7812754 , 0.5999229 ],
        [0.5057047 , 0.5719596 , 0.7552088 , 0.7380792 ]]])

for each_array in boxes[0]:
    print(each_array.shape)
    print(np.array([each_array]).shape)

scores = np.array([[9.8330945e-01, 9.4317716e-01]])

classes = np.array([[1.,1.]])

category_index = {1: {'id': 1, 'name': 'box'}}

to = TrackableObject(boxes, scores, classes, category_index)

assert to._boxes.shape == (1,2,4) , "Shape of boxes array is not (1,n,4)"

print(to._boxes.shape)
print("Success")
