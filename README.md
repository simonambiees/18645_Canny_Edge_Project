   
### 18645 Project Canny Edge Detector
## Optimized version of original C++ implementation of Canny Edge Detector
## Original C++ implementation from scratch of Canny Edge Detector on bmp images by YI SHI 2018 ys3237@nyu.edu
## Optimized version by Xiao Jin xjin2@andrew.cmu.edu
 
Structure
1. picIn    : where I store test images   
2. picOut	: output image directory      
3. source	: source code directory       
   -canny.cpp/h  : image processing   
   -util.cpp/h	: from scratch read and write bmp function  
   -main.cpp/h   : main   
4. test : test output history

Instruction:
```
Put your images in picIn folder. They must be bmp format.
Open util.h, modify H and W to your image's resolution.
Run "make -B"
Run "./canny <image_file> <number of benchmark runs>"

```
