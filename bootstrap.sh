#!/bin/bash

venvdir='venv'
unamestr=`uname`
wheelurl=''

if [[ "$unamestr" == 'Darwin' ]]; then
	#wheelurl='https://drive.google.com/open?id=0B9HTCbpQs6J1aHJ6a3gyRkhzTzA'
	wheelurl='tensorflow'
elif [[ "$unamestr" == "Linux" ]] ; then
	hostdomain=`hostname -d`
	if [[ "$hostdomain" == 'cs.utexas.edu' ]]; then
		#wheelurl='https://storage.googleapis.com/tensorflow_sourced/tensorflow-1.0.1-cp35-cp35m-linux_x86_64.whl'
		wheelurl='https://storage.googleapis.com/sparcit/lib/tensorflow/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl'
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
