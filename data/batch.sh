#!/bin/bash

for NQ in {21..25}
do
    for NC in {21..25}
    do
	for d in {2..6}
	do
	    echo "NQ:$NQ NC:$NC d:$d"
	    echo "NQ:$NQ NC:$NC d:$d" >> batch.txt
	    ./near --novalidation $NQ $NC $d >> batch.txt
	done
    done
done
echo All done
