default:
	g++ -mavx2 ./source/*.cpp -o canny.x

clean:
	rm -rf *~
	rm -rf *.x