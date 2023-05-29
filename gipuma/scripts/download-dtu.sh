#!/bin/bash
UNZIP_DIR="./data/dtu/"

function download_and_unzip() {
    TMPFILE=`mktemp`
    wget "$1" -O $TMPFILE
    unzip -d $UNZIP_DIR $TMPFILE
    rm $TMPFILE
}
mkdir -p $UNZIP_DIR

download_and_unzip http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
