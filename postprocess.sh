dos2unix hottiedata/output/positive.txt
dos2unix hottiedata/output/negative.txt
echo "Shifting predicted positive images to Images/sorted/positives..."
while IFS= read -r filename; do mv "$filename" Images/Sorted/positives; done < hottiedata/output/positive.txt
echo "Shifting predicted negative images to Images/sorted/negatives..."
while IFS= read -r filename; do mv "$filename" Images/Sorted/negatives; done < hottiedata/output/negative.txt
