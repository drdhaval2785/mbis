# Goal
To train a neural network to segregate good images and bad images.

# Programs used
python 2.7.9
octave 4.0.3

# Python libraries used
PIL - pip install pillow
resizeimage - pip install python-resize-image

# Folder structure
hottiedata
	input
		Place to store train / test data in CSV / txt format. (Nothing to be done by the user. Automatically generated.)
	output
		Place to store output data of false positives and false negatives on test data.
Images
	ImagesDir
		Here you will put your 'Good' images for training.
	JunkDir
		Here you will put your 'Bad' images for training.
	TestDir
		positives
			Here you will put your 'Good' test images for testing the neural network on unseen images.
		negatives
			Here you will put your 'Bad' test images for testing the neural network on unseen images.

# Steps
1. Put your images in the Images folder as shown above.
2. `cd path/to/this/folder'
3. Run `sh runcode.sh` from commandline.
4. This will generate all the text files / CSV files in hottiedata folder. (Extracted images and put their crux data in CSV / txt).
5. Step 1 to 3 are one time event. Unless you want to change images, you don't have to do this again.
6. Open Octave CLI
7. Write 'hottie' and press enter.
8. Keep on pushing enter whenever asked.
9. The console will show you the training / testing accuracy and also list of false positive / false negative files (places where the code failed).
10. Examine the image and try to figure out the cause of error.

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


