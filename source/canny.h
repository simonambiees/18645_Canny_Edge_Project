#ifndef CANNY_H
#define CANNY_H

#include "util.h"
#include <math.h>  
using namespace std;
// for guassian operator matrix generation
#define GAUS_SIZE 7
#define SIGMA 3
#define PI 3.14159265358979323846

//bool convol(int(&array)[H][W], double(&op)[GAUS_SIZE][GAUS_SIZE], int);
bool convol(int* array, int sizeImg[2], int* op, int sizeOp[2], int);
double convol_bench_wrapper(int(&array)[H][W]);
double convol_kernel_bench_wrapper(int(&array)[H][W]);
bool gausFilter(int(&array)[H][W]);
bool gradientForm(int(&array)[H][W],int);
bool gausFilter_SIMD(int(&array)[H][W]);
bool gradientForm_SIMD(int(&array)[H][W],int);
int angle_class(double);
bool nms(int(&magArray)[H][W], int(&dirArray)[H][W]);
bool thresHolding(int(&magArray)[H][W],bool, int);
int myMax(int, int, int);
bool fill(int(&array1)[H][W], int(&array2)[H][W],int div,int reduce);
bool histoBuild(int(&array)[H][W]);

extern int magGrad[H][W];
extern int magGradY[H][W];
extern int magGradX[H][W];
extern int magGradOut[H][W];
extern int dirGrad[H][W];
extern int magGradOut5[H][W];
extern int magGradOut3[H][W];
extern int magGradOut1[H][W];
extern int magGradOut3[H][W];
extern int countTable[256];
extern int oneDImgArray[H*W];
extern int tmpConvArray[H][W];

#endif
