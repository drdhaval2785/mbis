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
python resize.py Images/ImagesDir trues.csv trueset.txt
python resize.py Images/JunkDir falses.csv falseset.txt
cat hottiedata/input/falseset.txt hottiedata/input/trueset.txt > hottiedata/input/fullset.txt
python resize.py Images/TestDir/positives testpositives.csv testplus.txt
python resize.py Images/TestDir/negatives testnegatives.csv testminus.txt
python resize.py Images/ToBeSorted tobesorted.csv tosort.txt
