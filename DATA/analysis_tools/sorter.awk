#!/usr/bin/awk -f
BEGIN {

	numLine = 1
}
{

	if(numLine == ((ARGV[2]+1)*5)){
		print $19
	}
	#$16, 19

	if(numLine == 15){
		numLine = 1
	} else{
		numLine++
	}
}
END {

}

