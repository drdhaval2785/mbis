echo "Testing python version - The code is written and tested for python 2.7.9."
python --version
echo "Installing python image library - PIL..."
pip install pillow
echo "Installing python-resize-image library..."
pip install python-resize-image
echo "Preparing the folder structure..."
mkdir hottiedata
mkdir hottiedata/input
mkdir hottiedata/input/normalized
mkdir hottiedata/output
mkdir Images/ImagesDir
mkdir Images/JunkDir
mkdir Images/TestDir
mkdir Images/TestDir/positives
mkdir Images/TestDir/negatives
mkdir Images/ToBeSorted
mkdir Images/Sorted
mkdir Images/Sorted/positives
mkdir Images/Sorted/negatives
echo "Finished setting up"
echo "Next steps are -"
echo "1. Put training good images in Images/ImagesDir"
echo "2. Put training bad images in Images/JunkDir"
echo "3. Put the images to be separated in Images/ToBeSorted"
echo "4. Run sh preprocess.sh"
echo "5. Open Octave CLI"
echo "6. cd to the path of code in octave CLI"
echo "7. type hottie1 and press enter"
echo "8. run sh postprocess.sh"
echo "9. Check the folders Images/Sorted/positives and Images/Sorted/negatives for output."

