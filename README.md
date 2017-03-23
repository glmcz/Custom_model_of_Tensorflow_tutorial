# Custom_model_of_Tensorflow_tutorial
Conected with Android demo

There, I will try to describe 2 different aproach how to creat custom model for Android device.

1 Yolo model
If you like plain C and want the lightes model with best predictions and like to use boundaries, this is a way you should go.
https://pjreddie.com/darknet/yolo/
https://github.com/thtrieu/darkflow

2 TensorFlow model
TensorFlow is stil young project, but whit increadibly good community a rappid development. Therefore you should be aware 
on which version you working, not only tensorflow, but retrained models too.

With TensorFlow board you can vizualizate your graph, only with simple comand like tensorboard --logdir="path to checked-point.cpkt 
of your model".


I am using high level libraries for TensorFlow such like Keras and TFlearn. One disadvantage is exporting models to 
format readable for Android(for etc converting to Google protoBuff, .pb with libraries or customizable scripts). Another 
way is with Google Serving(https://tensorflow.github.io/serving/). 

You can find in my python scripts how to handle exporting keras model with or without TensorFlow Serving to .pb format.
