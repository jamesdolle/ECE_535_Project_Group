#!/usr/bin/awk -f
BEGIN {




	numLine = 0
}
{

	if(numLine == (ARGV[2])){
		print $8
	}

	if(numLine == 2){
		numLine = 0
	} else{
		numLine++
	}
}
END {

}

#grep "Class: 0" saveMe.txt > group.txt
#./sortG.awk group.txt 0/1/2
