#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
rm tmp -rf
mkdir tmp
for i in {1..250}
do
    nc -d localhost 8080 > tmp/$i.log &
done