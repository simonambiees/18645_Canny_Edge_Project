default:
	g++ -mavx2 -mfma -fopenmp -O1 ./source/*.cpp -o canny.x

clean:
	rm -rf *~
	rm -rf *.x