function W = rgbnormalize(X)
	mea = mean(X);
	W = (X-mea) ./ 256;
end
