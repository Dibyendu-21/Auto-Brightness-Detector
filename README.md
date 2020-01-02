# Auto-Brightness-Detector
Machine Learning model to detect the brightness of any image
## I/O
Input-Image, Output- Score of 0-10 (0 being lowest and 10 being highest)
## Dataset Used
UCLA protest dataset
## Pre-processing
For every image, the true brightness of the image was calculated as mean of all color channels in the image and a datset was created which bears the filename and the corresponding continuous brightness value. Final dataset was created by binning the continous brightness value (0-255) in 11 bins (0-10) which are the discrete brightness score any image can take. Final dataset had the imagename and dicsrete brightness score.

Dsicrete labels were converted to their one hot encode form. Train test split of the final dataset in 80-20 ratio was done to generate test set and a dummy set. This summy set was further splitted in the same ratio to generate the train and vaildation set.
## Training the model
A shallow CNN  model with a convolution base of 5 layers was used to train the model. Max pooling was employed to reduce the size of the activation map and also to reduce some noise. softmax function was used as the activation function at the output layer since output was selected out of 11 classes (0-10) and softmax generates a probaility distribution out of these 11 classes instead of probability score like in sigmoid. Loss funstion employed was categorical cross entropy and metric used was f-measure. The model was trained for 50 epochs using model checkpoint callback on the vaildation set to reduce overfitting.
## Evalaution
Model tested on unknown set yielded f-measure of 71%.
