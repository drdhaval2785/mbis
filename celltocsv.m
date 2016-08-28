function celltocsv (cellarray, csvfilename)
	fid=fopen(csvfilename,'w');
	x=cellarray;
	[rows,cols]=size(x);
	for i=1:rows
		  %fprintf(fid,'%s,',x{i,1:end-1})
		  %fprintf(fid,'%s\n',x{i,end})
		  fprintf(fid,'%s\n',x{i,1:end})
	end
	fclose(fid);
end
