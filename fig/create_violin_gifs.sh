#!/bin/bash

for dir in `ls -d *_violins`
do
    echo 'Creating gifs for' $dir
    convert -delay 175 -loop 0 $dir/*.png $dir/$dir\_all.gif
done
