# NN-Lib

## Contents

All the neccessary header files are in `/include` folder.
The source code is in `/source` folder.

* /include
	* matrix.h - Declaration of the Matrix class and all its functions
	* misc.h - Declaration of all miscellaneous classes and functions
	* network.h - Declaration of the Network class and all its functions

* /source
	* matrix.cpp - Matrix class implementation
	* misc.cpp - Implementation of miscellaneous classes and functions
	* network.cpp - Network class implementation

## Compiling

### Linux
To compile on Linux, all you have to do is make `compile.sh` executable with `chmod +x compile.sh` command. Then you can simply run `./main.out` and it should work.

### Windows
Have fun. \
\
These are the contents of `compile.sh` file: \
`g++ main.cpp source/misc.cpp source/matrix.cpp -o main.out`
