from PIL import Image
from resizeimage import resizeimage
import sys
from os import listdir
from os.path import isfile, join
def findfilenames(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles
def resize_file(in_file, out_file, size, filenamefile):
    with open(in_file,'r+b') as fd:
        im = Image.open(fd)
        image = resizeimage.resize('contain', im, size)
        pix = image.load()
        out = []
        for (a,b,c,d) in list(image.getdata()):
            out = out + [a] + [b] + [c] + [d]
    out = map(str,out)
    if len(out) == size[0] * size[1] * 4:
        fout = open(out_file, 'a')
        fout.write(','.join(out)+'\n')
    else:
        fout = open(out_file, 'a')
        fout.write(','.join(out)+'\n')
        print "Check", in_file, "- It doesn't seem to be in CYMK format."
    fout.close()
    filenamef = open(filenamefile,'a')
    filenamef.write(in_file+'\n')
    filenamef.close()

if __name__=="__main__":
    inputfolder = sys.argv[1]
    print "Reading the filenames from", inputfolder
    filenames = findfilenames(inputfolder)
    counter = 1
    print "Resizing and putting data into csv file."
    for member in filenames:
        if counter % 25 == 0:
            print counter, '...'
        counter += 1
        fin = inputfolder+'/'+member
        fout = 'hottiedata/input/'+sys.argv[2]
        filenamefile = 'hottiedata/input/'+sys.argv[3]
        resize_file(fin, fout, (20, 20), filenamefile)
	
