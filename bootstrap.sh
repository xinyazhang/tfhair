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
	#wheelurl='https://drive.google.com/open?id=0B9HTCbpQs6J1aHJ6a3gyRkhzTzA'
	wheelurl='tensorflow'
elif [[ "$unamestr" == "Linux" ]] ; then
	hostdomain=`hostname -d`
	avx2=`grep avx2 /proc/cpuinfo`
	if [[ "$hostdomain" == 'cs.utexas.edu' ]]; then
		avx2=`grep avx2 /proc/cpuinfo|wc -c`
		if [[ $avx2 -ne 0 ]]; then
			wheelurl='https://storage.googleapis.com/sparcit/lib/tensorflow/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl'
		fi
	fi
fi

if [[ "y$wheelurl" == 'y' ]]; then
	echo "bootstrap.sh cannot detect your system, probably your system already has TensorFlow installed by the administrator"
	echo "Exiting"
fi

echo "Creating virtualenv"
virtualenv --system-site-packages $venvdir
echo "virtualenv created"

source $venvdir/bin/activate
echo "virtualenv activated"

pip install --upgrade $wheelurl progressbar2

echo "TensorFlow installed under virtualenv directory $venvdir"
echo "Leaving"
