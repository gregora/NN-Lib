#build main.out
main.out: bin/nnlib.so main.cpp
	g++ -o main.out main.cpp bin/nnlib.so -lpthread

#build shared library and move it to bin/ folder
bin/nnlib.so: matrix.o misc.o network.o algorithms.o
	ld -relocatable algorithms.o network.o misc.o matrix.o -o nnlib.so
	rm matrix.o misc.o network.o algorithms.o
	mv nnlib.so bin/nnlib.so


#build object files

algorithms.o: include/algorithms.h source/algorithms.cpp include/matrix.h include/misc.h include/network.h
	g++ -c source/algorithms.cpp -Wall -pedantic

network.o: include/network.h source/network.cpp
	g++ -c source/network.cpp -Wall -pedantic

misc.o: include/misc.h source/misc.cpp
	g++ -c source/misc.cpp -Wall -pedantic

matrix.o: include/matrix.h source/matrix.cpp
	g++ -c source/matrix.cpp -Wall -pedantic
