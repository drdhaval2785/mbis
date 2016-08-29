echo "Prepare the folder structure."
rm -rf hottiedata
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
echo "Resize the images in Images/ImagesDir directory and store in hottiedata/input/trues.csv..."
python resize.py Images/ImagesDir trues.csv trueset.txt
echo "Resize the images in Images/JunkDir directory and store in hottiedata/input/falses.csv..."
python resize.py Images/JunkDir falses.csv falseset.txt
echo "Resize the images in Images/TestDir/positives directory and store in hottiedata/input/testpositives.csv..."
python resize.py Images/TestDir/positives testpositives.csv testplus.txt
echo "Resize the images in Images/TestDir/negatives directory and store in hottiedata/input/testnegatives.csv..."
python resize.py Images/TestDir/negatives testnegatives.csv testminus.txt
echo "Resize the images in Images/ToBeSorted directory and store in hottiedata/input/tobesorted.csv..."
python resize.py Images/ToBeSorted tobesorted.csv tosort.txt

echo "Finished preprocessing"
echo "Next steps are -"
echo "1. Open Octave CLI"
echo "2. `cd` to the path of code in octave CLI"
echo "3. type `hottie1` and press enter"
echo "4. run `sh postprocess.sh`"
echo "5. Check the folders Images/Sorted/positives and Images/Sorted/negatives for output."
