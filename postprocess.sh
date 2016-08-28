dos2unix hottiedata/output/positive.txt
dos2unix hottiedata/output/negative.txt
while IFS= read -r filename; do mv "$filename" Images/Sorted/positives; done < hottiedata/output/positive.txt
while IFS= read -r filename; do mv "$filename" Images/Sorted/negatives; done < hottiedata/output/negative.txt
