# Goal
To train a neural network to segregate good images and bad images.

# Programs used
1. python >= 2.7.9 https://www.python.org/downloads/
2. octave 4.0.3 - https://www.gnu.org/software/octave/download.html
3. [git bash](https://git-scm.com/downloads) / any unix / mac system (To run shell files)

# Setup (One time event)
0. `cd path/to/this/folder'
1. `sh setup.sh`. - This is a one time event. This will install python packages needed for functioning and make folder structure.

# Python libraries used (Done via setup.sh)
1. PIL - pip install pillow
2. resizeimage - pip install python-resize-image

# Folder structure (Done via setup.sh)
1 Images (User has to put images)

* ImagesDir - Here you will put your 'Good' images for training.
* JunkDir - Here you will put your 'Bad' images for training.
* ToBeSorted - Here you will put your new images - to be classified.

2 hottiedata (Nothing to be done by user. Auto-generated)

* input - Place to store train / test data in CSV / txt format. (Nothing to be done by the user. Automatically generated.)
* output - Place to store output data of false positives and false negatives on test data.

# Steps for usage
1. `cd path/to/this/folder'
2. Put your images in the Images folder as shown in the `Images` folder description. The description is self explanatory.
3. Run `sh preprocess.sh` from commandline.
4. This will generate all the text files / CSV files in hottiedata folder. (Extracted images and put their crux data in CSV / txt).
5. Step 1 to 3 are one time event. Unless you want to change images, you don't have to do this again.
6. Open Octave CLI.
7. write `cd path/to/this/folder' and click enter
8. Write 'hottie1' and press enter.
9. Keep on pushing enter whenever asked.
10. The console will show you the training / testing accuracy and also list of false positive / false negative files (places where the code failed).
11. Examine the image and try to figure out the cause of error.
12. run `sh postprocess.sh` from commandline. This will segregate the good and bad images from Images/ToBeSorted folder into Images/sorted/positives and Images/sorted/negatives folders.
13. Check the Images/sorted/positives and Images/sorted/negatives for segregated images.

# Developer notes
The next sections are for developer notes. Ignore if you are a user.

# Preprocessing
1. `resize.py` does the preprocessing.
2. Images are usually of different sizes. Therefore, it is necessary that they are converted to uniform data for machine learning.
3. We resize every image into 10X10 pixel size (i.e. make it very low resolution).
4. Every pixel has 4 features CYMK which take values from 0-255 (i.e. 256 possible values).
5. Therefore our input data has 10X10X4 = 400 input features.
6. These 400 input features are written in a CSV file. One line represents one image file.
7. All files from the input folders are converted to the corresponding CSV files by this process.
8. The file names are also stored as txt files, for later retrieval (which line of data belongs to which file).
9. After preprocessing, we don't use images any more. We do our training etc with the CSV files only.

# Normalization
1. As mentoned in preprocessing section, our data has values from 0-255, and a total of 400 columns for each image.
2. If we keep the numbers as they are, it would be difficult to train neural networks. Therefore, we need to normalize data.
3. Python can normalize, but as the data size is higher, a Linear Algebra program like octave does it faster. So I have chosen octave to do this.
3. We take average `mean` for each column and normalize according to the following formula `(observation-mean)/256`. Note that mean varies for each column.
4. Thus, each value is converted to (-1,1) range.
5. If the input images are not changed, the normalized values would also not change. Therefore, we have stored the normalized data in hottiedata/input/normalized/traindata.mat.
6. This will ensure that the data is loaded directly from this file as long as there is such a file.
7. This will help us run the octave code for training neural network a bit faster.
8. If you change the images, please delete the traindata.mat file.
9. Code is resilient enough and will regenerate a new testdata.mat with the new images you have put in folders.

# Neural Network Parameters
1. Input layer - 400 (Input features of pixels)
2. Hidden layer - 50
3. Output layer - 2 (Yes / No)
4. Iterations - 100
5. lambda - 1

# Training Neural Network
1. We train the neural network with the training data.
2. We display training classification accuracy.
3. We display test classification accuracy.
4. If the accuracy on test classification is 80% (or greater than some user defined threshold), the parameters are stored in hottiedata/input/learntparameters.mat.
5. This way we can store the neural network we found OK, and then apply it to some other test cases too.

# Acknowledgements
1. Code of Coursera Machine learning course ex4 has been extensively used to create neural network on octave.
