#!/bin/bash

tensorinfo=`pip2 show tensorflow|wc -c`

if [[ $tensorinfo -ne 0 ]]; then
	echo "This system already has tensorflow installed for python2"
	echo "There is no need to run this script"
	echo "Exiting"
	exit
fi

venvdir='venv'
unamestr=`uname`
wheelurl=''

if [[ "$unamestr" == 'Darwin' ]]; then
	# FIXME: check CPU info and use optimized wheel
	wheelurl='tensorflow'
elif [[ "$unamestr" == "Linux" ]] ; then
	avx2=`grep avx2 /proc/cpuinfo|wc -c`
	if [[ $avx2 -ne 0 ]]; then
		wheelurl='https://storage.googleapis.com/sparcit/lib/tensorflow/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl'
	fi
fi

if [[ "y$wheelurl" == 'y' ]]; then
	echo "bootstrap.sh cannot determine the tensorflow package for your system"
	echo "Using the official tensorflow package from pip"
	wheelurl='tensorflow'
fi

echo "Creating virtualenv"
virtualenv --system-site-packages $venvdir
echo "virtualenv created"

source $venvdir/bin/activate
echo "virtualenv activated"

pip install --upgrade $wheelurl progressbar2

echo "TensorFlow installed under virtualenv directory $venvdir"
echo "Leaving"
