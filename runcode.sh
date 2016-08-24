rm -rf Data
mkdir Data
python resize.py Images/ImagesDir trues.csv trueset.txt
python resize.py Images/JunkDir falses.csv falseset.txt
python resize.py Images/TestDir/positives testpositives.csv testplus.txt
python resize.py Images/TestDir/negatives testnegatives.csv testminus.txt
