//canny_kernel.cpp

#include "canny.h"
#include "main.h"
#include <immintrin.h> 
#include <cstddef>
#include "rdtsc.h"
#include <omp.h>

//global 2d array for writing
int magGrad[H][W];
int magGradX[H][W];
int magGradY[H][W];
int dirGrad[H][W];
int tmpConvArray[H][W];
int magGradOut[H][W];
int magGradOut5[H][W];
int magGradOut3[H][W];
int magGradOut1[H][W];

int countTable[256] = { 0 };
int OP_PEWITT_X[9] = { -1,0,1,-1,0,1,-1,0,1 };
int OP_PEWITT_Y[9] = { 1,1,1,0,0,0,-1,-1,-1 };
int OP_SOBEL_X[9] = { -1,0,1,-2,0,2,-1,0,1 };
int OP_SOBEL_Y[9] = { 1,2,1,0,0,0,-1,-2,-1 };
int givenGausFil[49] = {  1,1,2,2,2,1,1, 1,2,2,4,2,2,1, 2,2,4,8,4,2,2 ,2,4,8,16,8,4,2 , 2,2,4,8,4,2,2 , 1,2,2,4,2,2,1 , 1,1,2,2,2,1,1  };
int oneDImgArray[H*W] = { 0 };
int removed = 0;
// Function convolution  array: image, sizeArray:size of image
//                       op: operator, sizeOp:size of operator
bool convol(int* array, int sizeArray[2], int* op, int sizeOp[2],int stride)
{
	size_t rowArray = sizeArray[0];
	size_t colArray = sizeArray[1];
	size_t rowOp = sizeOp[0];
	size_t colOp = sizeOp[1];

	for (int i = 0; i < rowArray ; i = i + stride)
	{
		for (int j = 0; j < colArray ; j = j + stride)
		{
            // use a temporary global array to transfer data
            // not a optimum way but working.
			tmpConvArray[i][j] = 0;
			if (i <rowOp / 2 || i >= rowArray - rowOp / 2 || j <colOp / 2 || j >= colArray - colOp / 2)
			{
				continue;
			}
			for (int p = i - rowOp / 2; p < i + rowOp / 2+1; p++)
			{
				for (int q = j - colOp / 2; q < j + colOp / 2+1; q++)
				{
					tmpConvArray [i][j] += array[p * colArray + q] * op[(p - (i - rowOp / 2)) * colOp + (q - (j - colOp / 2))];
				}
			}
		}
	}
	
	return 1;
}

double convol_benchmark(int* array, int sizeArray[2], int* op, int sizeOp[2],int stride)
{
	size_t rowArray = sizeArray[0];
	size_t colArray = sizeArray[1];
	size_t rowOp = sizeOp[0];
	size_t colOp = sizeOp[1];

	tsc_counter t0, t1;
	long long sum_cycle = 0;
	long long sum_inst = 0;
	double cycles = 0.0;

	RDTSC(t0);
	for (int i = 0; i < rowArray ; i = i + stride)
	{
		for (int j = 0; j < colArray ; j = j + stride)
		{
            // use a temporary global array to transfer data
            // not a optimum way but working.
			tmpConvArray[i][j] = 0;
			if (i <rowOp / 2 || i >= rowArray - rowOp / 2 || j <colOp / 2 || j >= colArray - colOp / 2)
			{
				continue;
			}
			for (int p = i - rowOp / 2; p < i + rowOp / 2+1; p++)
			{
				for (int q = j - colOp / 2; q < j + colOp / 2+1; q++)
				{
					tmpConvArray [i][j] += array[p * colArray + q] * op[(p - (i - rowOp / 2)) * colOp + (q - (j - colOp / 2))];
				}
			}
		}
	}
	RDTSC(t1);
	sum_cycle += (long long)COUNTER_DIFF(t1, t0, CYCLES);
	
	// return cpu cycles
	cycles = (double)sum_cycle;
	return cycles;
}

double convol_bench_wrapper(int(&array)[H][W])
{
	int sizeImg[2] = {H,W};
	int sizeOp[2] = {GAUS_SIZE,GAUS_SIZE};
	double cycles = 0.0;

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			oneDImgArray[i * W + j] = array[i][j] ;
		}
	}
	cycles = convol_benchmark(oneDImgArray, sizeImg, givenGausFil, sizeOp,1);
	return cycles;
}

bool convol_optimized(double* array, int sizeArray[2], double* op, int sizeOp[2],int stride)
{
	// i,j: output image index
	// m,n: kernel index
	// p,q: input image index
	// stride is always 1 in this case

	tsc_counter t0, t1;
	size_t rowArray = sizeArray[0];
	size_t colArray = sizeArray[1];
	size_t rowOp = sizeOp[0];
	size_t colOp = sizeOp[1];
	size_t rowOpHalf = rowOp / 2;
	size_t colOpHalf = colOp / 2;
	size_t rowOutput = rowArray - 2 * rowOpHalf;
	size_t colOutput = colArray - 2 * colOpHalf;
	// printf("rowArray: %d, colArray: %d\n", rowArray, colArray);
	// printf("rowOp: %d, colOp: %d\n", rowOp, colOp);
	// printf("rowOutput: %d, colOutput: %d\n", rowOutput, colOutput);

	// zero-pad the input image so that the output col_size is a multiple of 40
	// this is to make the SIMD implementation easier
	// TODO
	// Or choose image size wisely to avoid padding
	// make one output col_size 40k, where k is an integer
	// the other output col_size 40k + 4
	// image size 4006x3024
	// test image size 46x30

	// Initialize the output image to zero
	double* output = new double[rowOutput * colOutput]();


	// Loop over every pixel in the output image
	// calculating 40 outputs at a time
	for (int i=0; i<rowOutput; i++) { // 1 row at a time
		for (int j=0; j<colOutput/40*40; j=j+40) { // 40 columns at a time
			// Maybe initialize the output to zero here?
			// Load (40/4 = 10) SIMD registers for output image
			// Load the output image chunk of 40 pixels at a time
			__m256d sum_0 = _mm256_loadu_pd(&output[i * colOutput + j]);
			__m256d sum_1 = _mm256_loadu_pd(&output[i * colOutput + j + 4]);
			__m256d sum_2 = _mm256_loadu_pd(&output[i * colOutput + j + 8]);
			__m256d sum_3 = _mm256_loadu_pd(&output[i * colOutput + j + 12]);
			__m256d sum_4 = _mm256_loadu_pd(&output[i * colOutput + j + 16]);
			__m256d sum_5 = _mm256_loadu_pd(&output[i * colOutput + j + 20]);
			__m256d sum_6 = _mm256_loadu_pd(&output[i * colOutput + j + 24]);
			__m256d sum_7 = _mm256_loadu_pd(&output[i * colOutput + j + 28]);
			__m256d sum_8 = _mm256_loadu_pd(&output[i * colOutput + j + 32]);
			__m256d sum_9 = _mm256_loadu_pd(&output[i * colOutput + j + 36]);

			// Loop over every pixel in the kernel
			for (int m=0; m<rowOp; m++) {
				for (int n=0; n<colOp; n++) {
					// Calculate the start of the chunk of the input image
					// that corresponds to the kernel pixel
					int p = i + m;
					int q = j + n;

					// Load the kernel value
					// broad cast the kernel value to all elements in the SIMD register
					__m256d kernel = _mm256_set1_pd(op[m * colOp + n]);

					// Load the input image chunk of 4 pixels at a time
					// convert input image value from int to double
					// load 4 pixels into a SIMD register
					__m256d input_a = _mm256_loadu_pd(&array[p * colArray + q]);
					__m256d input_b = _mm256_loadu_pd(&array[p * colArray + q + 4]);
					__m256d input_c = _mm256_loadu_pd(&array[p * colArray + q + 8]);
					__m256d input_d = _mm256_loadu_pd(&array[p * colArray + q + 12]);
					__m256d input_e = _mm256_loadu_pd(&array[p * colArray + q + 16]);
					sum_0 = _mm256_fmadd_pd(input_a, kernel, sum_0);
					sum_1 = _mm256_fmadd_pd(input_b, kernel, sum_1);
					sum_2 = _mm256_fmadd_pd(input_c, kernel, sum_2);
					sum_3 = _mm256_fmadd_pd(input_d, kernel, sum_3);
					sum_4 = _mm256_fmadd_pd(input_e, kernel, sum_4);
					input_a = _mm256_loadu_pd(&array[p * colArray + q + 20]);
					input_b = _mm256_loadu_pd(&array[p * colArray + q + 24]);
					input_c = _mm256_loadu_pd(&array[p * colArray + q + 28]);
					input_d = _mm256_loadu_pd(&array[p * colArray + q + 32]);
					input_e = _mm256_loadu_pd(&array[p * colArray + q + 36]);
					sum_5 = _mm256_fmadd_pd(input_a, kernel, sum_5);
					sum_6 = _mm256_fmadd_pd(input_b, kernel, sum_6);
					sum_7 = _mm256_fmadd_pd(input_c, kernel, sum_7);
					sum_8 = _mm256_fmadd_pd(input_d, kernel, sum_8);
					sum_9 = _mm256_fmadd_pd(input_e, kernel, sum_9);

					// Store results back into the output image
					_mm256_storeu_pd(&output[i * colOutput + j], sum_0);
					_mm256_storeu_pd(&output[i * colOutput + j + 4], sum_1);
					_mm256_storeu_pd(&output[i * colOutput + j + 8], sum_2);
					_mm256_storeu_pd(&output[i * colOutput + j + 12], sum_3);
					_mm256_storeu_pd(&output[i * colOutput + j + 16], sum_4);
					_mm256_storeu_pd(&output[i * colOutput + j + 20], sum_5);
					_mm256_storeu_pd(&output[i * colOutput + j + 24], sum_6);
					_mm256_storeu_pd(&output[i * colOutput + j + 28], sum_7);
					_mm256_storeu_pd(&output[i * colOutput + j + 32], sum_8);
					_mm256_storeu_pd(&output[i * colOutput + j + 36], sum_9);
					
				}
			}
		}

		// Handle remaining elements if colOp is not divisible by 40
		if (colOutput % 40 != 0) {
			for (int j = colOutput - colOutput % 40; j < colOutput; j++) {
				for (int m = 0; m < rowOp; m++) {
					for (int n = 0; n < colOp; n++) {
						output[i * colOutput + j] += array[(i + m) * colArray + (j + n)] * op[m * colOp + n];
					}
				}
			}
		}
	}

	RDTSC(t0);
	// Copy the output image back to the global tmp array
	#pragma omp parallel for collapse(2) num_threads(4)
	for (int i=0; i<rowArray; i++) {
		for (int j=0; j<colArray; j++) {
			if (i < rowOpHalf || i >= rowArray - rowOpHalf || j < colOpHalf || j >= colArray - colOpHalf) {
				tmpConvArray[i][j] = 0;
			}
			else {
				tmpConvArray[i][j] = static_cast<int>(output[(i-rowOpHalf) * colOutput + (j-colOpHalf)]);
			}
		}
	}
	RDTSC(t1);
	int_conversion_cycles_counter += (double)COUNTER_DIFF(t1, t0, CYCLES);

	return 1;
}


// Converts image and op to doubles 
// Wrapper function for convol_optimized
bool convol_kernel(int* array, int sizeArray[2], int* op, int sizeOp[2],int stride)
{
	tsc_counter t0, t1;
	double* inputImgArray = new double[sizeArray[0] * sizeArray[1]];
	double inputOpArray[sizeOp[0] * sizeOp[1]];
	double cycles = 0.0;
	
	RDTSC(t0);
	#pragma omp parallel for collapse(2) num_threads(4)
	for (int i = 0; i < sizeArray[0]; i++)
	{
		for (int j = 0; j < sizeArray[1]; j++)
		{
			inputImgArray[i * sizeArray[1] + j] = static_cast<double>(array[i * sizeArray[1] + j]) ;
		}
	}

	
	#pragma omp parallel for collapse(2) num_threads(4)
	for (int i = 0; i < sizeOp[0]; i++)
	{
		for (int j = 0; j < sizeOp[1]; j++)
		{
			inputOpArray[i * sizeOp[1] + j] = static_cast<double>(givenGausFil[i * sizeOp[1] + j]);
		}
	}
	RDTSC(t1);
	int_conversion_cycles_counter += (double)COUNTER_DIFF(t1, t0, CYCLES);
	// printf("int_conversion_cycles_counter: %f\n", int_conversion_cycles_counter);
	bool result = convol_optimized(inputImgArray, sizeArray, inputOpArray, sizeOp, stride);
	delete[] inputImgArray;
	return result;
}


double convol_kernel_benchmark(double* array, int sizeArray[2], double* op, int sizeOp[2],int stride)
{
	// i,j: output image index
	// m,n: kernel index
	// p,q: input image index
	// stride is always 1 in this case

	size_t rowArray = sizeArray[0];
	size_t colArray = sizeArray[1];
	size_t rowOp = sizeOp[0];
	size_t colOp = sizeOp[1];
	size_t rowOpHalf = rowOp / 2;
	size_t colOpHalf = colOp / 2;
	size_t rowOutput = rowArray - 2 * rowOpHalf;
	size_t colOutput = colArray - 2 * colOpHalf;

	tsc_counter t0, t1;
	long long sum_cycle = 0;
	long long sum_inst = 0;
	double cycles = 0.0;

	// zero-pad the input image so that the output col_size is a multiple of 40
	// this is to make the SIMD implementation easier
	// TODO
	// Or choose image size wisely to avoid padding
	// make one output col_size 40k, where k is an integer
	// the other output col_size 40k + 4
	// image size 4006x3024
	// test image size 46x30

	// Initialize the output image to zero
	double* output = new double[rowOutput * colOutput]();

	RDTSC(t0);
	// Loop over every pixel in the output image
	// calculating 40 outputs at a time
	for (int i=0; i<rowOutput; i++) { // 1 row at a time
		for (int j=0; j<colOutput/40*40; j=j+40) { // 40 columns at a time
			// Maybe initialize the output to zero here?
			// Load (40/4 = 10) SIMD registers for output image
			// Load the output image chunk of 40 pixels at a time
			__m256d sum_0 = _mm256_loadu_pd(&output[i * colOutput + j]);
			__m256d sum_1 = _mm256_loadu_pd(&output[i * colOutput + j + 4]);
			__m256d sum_2 = _mm256_loadu_pd(&output[i * colOutput + j + 8]);
			__m256d sum_3 = _mm256_loadu_pd(&output[i * colOutput + j + 12]);
			__m256d sum_4 = _mm256_loadu_pd(&output[i * colOutput + j + 16]);
			__m256d sum_5 = _mm256_loadu_pd(&output[i * colOutput + j + 20]);
			__m256d sum_6 = _mm256_loadu_pd(&output[i * colOutput + j + 24]);
			__m256d sum_7 = _mm256_loadu_pd(&output[i * colOutput + j + 28]);
			__m256d sum_8 = _mm256_loadu_pd(&output[i * colOutput + j + 32]);
			__m256d sum_9 = _mm256_loadu_pd(&output[i * colOutput + j + 36]);

			// Loop over every pixel in the kernel
			for (int m=0; m<rowOp; m++) {
				for (int n=0; n<colOp; n++) {
					// Calculate the start of the chunk of the input image
					// that corresponds to the kernel pixel
					int p = i + m;
					int q = j + n;

					// Load the kernel value
					// broad cast the kernel value to all elements in the SIMD register
					__m256d kernel = _mm256_set1_pd(op[m * colOp + n]);

					// Load the input image chunk of 4 pixels at a time
					// convert input image value from int to double
					// load 4 pixels into a SIMD register
					__m256d input_a = _mm256_loadu_pd(&array[p * colArray + q]);
					__m256d input_b = _mm256_loadu_pd(&array[p * colArray + q + 4]);
					__m256d input_c = _mm256_loadu_pd(&array[p * colArray + q + 8]);
					__m256d input_d = _mm256_loadu_pd(&array[p * colArray + q + 12]);
					__m256d input_e = _mm256_loadu_pd(&array[p * colArray + q + 16]);
					sum_0 = _mm256_fmadd_pd(input_a, kernel, sum_0);
					sum_1 = _mm256_fmadd_pd(input_b, kernel, sum_1);
					sum_2 = _mm256_fmadd_pd(input_c, kernel, sum_2);
					sum_3 = _mm256_fmadd_pd(input_d, kernel, sum_3);
					sum_4 = _mm256_fmadd_pd(input_e, kernel, sum_4);
					input_a = _mm256_loadu_pd(&array[p * colArray + q + 20]);
					input_b = _mm256_loadu_pd(&array[p * colArray + q + 24]);
					input_c = _mm256_loadu_pd(&array[p * colArray + q + 28]);
					input_d = _mm256_loadu_pd(&array[p * colArray + q + 32]);
					input_e = _mm256_loadu_pd(&array[p * colArray + q + 36]);
					sum_5 = _mm256_fmadd_pd(input_a, kernel, sum_5);
					sum_6 = _mm256_fmadd_pd(input_b, kernel, sum_6);
					sum_7 = _mm256_fmadd_pd(input_c, kernel, sum_7);
					sum_8 = _mm256_fmadd_pd(input_d, kernel, sum_8);
					sum_9 = _mm256_fmadd_pd(input_e, kernel, sum_9);

					// Store results back into the output image
					_mm256_storeu_pd(&output[i * colOutput + j], sum_0);
					_mm256_storeu_pd(&output[i * colOutput + j + 4], sum_1);
					_mm256_storeu_pd(&output[i * colOutput + j + 8], sum_2);
					_mm256_storeu_pd(&output[i * colOutput + j + 12], sum_3);
					_mm256_storeu_pd(&output[i * colOutput + j + 16], sum_4);
					_mm256_storeu_pd(&output[i * colOutput + j + 20], sum_5);
					_mm256_storeu_pd(&output[i * colOutput + j + 24], sum_6);
					_mm256_storeu_pd(&output[i * colOutput + j + 28], sum_7);
					_mm256_storeu_pd(&output[i * colOutput + j + 32], sum_8);
					_mm256_storeu_pd(&output[i * colOutput + j + 36], sum_9);
					
				}
			}
		}

		// Handle remaining elements if colOp is not divisible by 40
		if (colOutput % 40 != 0) {
			for (int j = colOutput - colOutput % 40; j < colOutput; j++) {
				for (int m = 0; m < rowOp; m++) {
					for (int n = 0; n < colOp; n++) {
						output[i * colOutput + j] += array[(i + m) * colArray + (j + n)] * op[m * colOp + n];
					}
				}
			}
		}
	}
	RDTSC(t1);
	sum_cycle += (long long)COUNTER_DIFF(t1, t0, CYCLES);

	// Copy the output image back to the global tmp array
	for (int i=0; i<rowArray; i++) {
		for (int j=0; j<colArray; j++) {
			if (i < rowOpHalf || i >= rowArray - rowOpHalf || j < colOpHalf || j >= colArray - colOpHalf) {
				tmpConvArray[i][j] = 0;
			}
			else {
				tmpConvArray[i][j] = output[(i-rowOpHalf) * colOutput + (j-colOpHalf)];
			}
		}
	}

	// return cpu cycles
	cycles = (double)sum_cycle;
	return cycles;
}

double convol_kernel_bench_wrapper(int(&array)[H][W])
{
	int sizeImg[2] = {H,W};
	int sizeOp[2] = {GAUS_SIZE,GAUS_SIZE};
	double* inputImgArray = new double[H * W];
	double doubleGausFil[GAUS_SIZE*GAUS_SIZE];
	double cycles = 0.0;

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			inputImgArray[i * W + j] = static_cast<double>(array[i][j]) ;
		}
	}
	for (int i = 0; i < GAUS_SIZE; i++)
	{
		for (int j = 0; j < GAUS_SIZE; j++)
		{
			doubleGausFil[i * GAUS_SIZE + j] = static_cast<double>(givenGausFil[i * GAUS_SIZE + j]);
		}
	}
	cycles = convol_kernel_benchmark(inputImgArray, sizeImg, doubleGausFil, sizeOp,1);
	delete[] inputImgArray;
	return cycles;
}



// for guassian filter, may blur the image a bit but improve our overall edge detecting effect
bool gausFilter(int(&array)[H][W])
{
	// printf("rowArray: %d, colArray: %d\n", H, W);
	int sizeImg[2] = {H,W};
	int sizeOp[2] = {GAUS_SIZE,GAUS_SIZE};
	// cout << "start gaussian filtering:" <<endl;
    
    //since my convolution fucntion code was done in a 1d array
    // transform 2d to 1 d
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			oneDImgArray[i * W + j] = array[i][j] ;
		}
	}
    // printf("rowop: %d, colop: %d\n", sizeOp[0], sizeOp[1]);
	convol(oneDImgArray, sizeImg, givenGausFil, sizeOp,1);
	fill_opt(tmpConvArray, img, 140, 3);
	return 1;
}

bool gausFilter_opt(int(&array)[H][W])
{
	// printf("rowArray: %d, colArray: %d\n", H, W);
	int sizeImg[2] = {H,W};
	int sizeOp[2] = {GAUS_SIZE,GAUS_SIZE};
	// cout << "start gaussian filtering:" <<endl;
    
    //since my convolution fucntion code was done in a 1d array
    // transform 2d to 1 d
	#pragma omp parallel for collapse(2) num_threads(4)
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			oneDImgArray[i * W + j] = array[i][j] ;
		}
	}
    // printf("rowop: %d, colop: %d\n", sizeOp[0], sizeOp[1]);
	convol_kernel(oneDImgArray, sizeImg, givenGausFil, sizeOp,1);
	fill_opt(tmpConvArray, img, 140, 3);
	return 1;
}

//
bool gradientForm(int(&array)[H][W],int opType)
{
  	int rowArray = H;
	int colArray = W;
	// printf("rowArray: %d, colArray: %d\n", rowArray, colArray);
	int xGrad = 0;
	int yGrad = 0;
	if (opType == 0)
    { //naive way for gradient magnitude computation Gy = y[j]-y[j-1]
        for (int i = 1; i < rowArray; i++)
		{
			for (int j = 1; j < colArray; j++)
			{
				//int sizeOp[2] = { GAUS_SIZE,GAUS_SIZE };
				xGrad = array[i][j] - array[i][j - 1];
				yGrad = array[i][j] - array[i - 1][j];
				magGradY[i][j] = yGrad;
				magGradX[i][j] = xGrad;
				magGrad[i][j] = sqrt(yGrad * yGrad + xGrad * xGrad);
				dirGrad[i][j] = angle_class(atan2(yGrad, xGrad) / PI * 180);
			}
		}
	}
	else if (opType == 1)
	{
        // We use pewitt operator and convlution to do gradient computing
		int sizeImg[2] = { H,W };
		int sizeOp[2] = { 3,3 };
        //convolution
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				oneDImgArray[i*W + j] = array[i][j];
			}
		}
        //Gx
		// printf("rowop: %d, colop: %d\n", sizeOp[0], sizeOp[1]);
		convol(oneDImgArray, sizeImg, OP_PEWITT_X, sizeOp, 1);
		fill_opt(tmpConvArray, magGradX,1,4);
        //Gy
		convol(oneDImgArray, sizeImg, OP_PEWITT_Y, sizeOp, 1);
		fill_opt(tmpConvArray, magGradY,1,4);
        
        // make sure no pixel has a value larger than 255 (maximum of our greyscale)
		for (int i = 0; i < rowArray; i++)
		{
			for (int j = 0; j < colArray; j++)
			{
				if (abs(magGradX[i][j]) + abs(magGradY[i][j]) > 255) {
					magGrad[i][j] = 255;
				}
				else 
				{
                    //magnitude
					magGrad[i][j] = abs(magGradX[i][j]) + abs(magGradY[i][j]);	
				}
                //direction
				dirGrad[i][j] = angle_class(atan2(magGradY[i][j], magGradX[i][j]) / PI * 180);
			}
		}
    }
	return 1;
}

bool gradientForm_opt(int(&array)[H][W],int opType)
{
  	int rowArray = H;
	int colArray = W;
	// printf("rowArray: %d, colArray: %d\n", rowArray, colArray);
	int xGrad = 0;
	int yGrad = 0;
	if (opType == 0)
    { //naive way for gradient magnitude computation Gy = y[j]-y[j-1]
	#pragma omp parallel for collapse(2) num_threads(4)
        for (int i = 1; i < rowArray; i++)
		{
			for (int j = 1; j < colArray; j++)
			{
				//int sizeOp[2] = { GAUS_SIZE,GAUS_SIZE };
				xGrad = array[i][j] - array[i][j - 1];
				yGrad = array[i][j] - array[i - 1][j];
				magGradY[i][j] = yGrad;
				magGradX[i][j] = xGrad;
				magGrad[i][j] = sqrt(yGrad * yGrad + xGrad * xGrad);
				dirGrad[i][j] = angle_class(atan2(yGrad, xGrad) / PI * 180);
			}
		}
	}
	else if (opType == 1)
	{
        // We use pewitt operator and convlution to do gradient computing
		int sizeImg[2] = { H,W };
		int sizeOp[2] = { 3,3 };
        //convolution
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				oneDImgArray[i*W + j] = array[i][j];
			}
		}
        //Gx
		// printf("rowop: %d, colop: %d\n", sizeOp[0], sizeOp[1]);
		convol_kernel(oneDImgArray, sizeImg, OP_PEWITT_X, sizeOp, 1);
		fill_opt(tmpConvArray, magGradX,1,4);
        //Gy
		convol_kernel(oneDImgArray, sizeImg, OP_PEWITT_Y, sizeOp, 1);
		fill_opt(tmpConvArray, magGradY,1,4);
        
        // make sure no pixel has a value larger than 255 (maximum of our greyscale)
		#pragma omp parallel for collapse(2) num_threads(4)
		for (int i = 0; i < rowArray; i++)
		{
			for (int j = 0; j < colArray; j++)
			{
				if (abs(magGradX[i][j]) + abs(magGradY[i][j]) > 255) {
					magGrad[i][j] = 255;
				}
				else 
				{
                    //magnitude
					magGrad[i][j] = abs(magGradX[i][j]) + abs(magGradY[i][j]);	
				}
                //direction
				dirGrad[i][j] = angle_class(atan2(magGradY[i][j], magGradX[i][j]) / PI * 180);
			}
		}
    }
	return 1;
}

//classify the angle into 4 classes
int angle_class(double angle)
{
	if ((angle < 22.5 && angle >= -22.5) || angle >= 157.5 || angle < -157.5)
	{
		return 0;
	}
	else if ((angle >= 22.5 && angle < 67.5) || (angle < -112.5 && angle >= -157.5))
	{
		return 1;
	}
	else if ((angle >= 67.5 && angle < 112.5) || (angle < -67.5 && angle >= -112.5))
	{
		return 2;
	}
	else if ((angle >= 112.5 && angle < 157.5) || (angle < -22.5 && angle >= -67.5))
	{
		return 3;
	}
	else
	{
		return 1;
	}
}

//Function: non maximum suppression
//all those trivial 'if' statement are for edge and corner cases
bool nms(int(&magArray)[H][W],int(&dirArray)[H][W])
{
	
	size_t rowArray = H;
	size_t colArray = W;
	for (int i = 1; i < rowArray; i++)
	{
		for (int j = 1; j < colArray; j++)
		{
			switch (dirArray[i][j]) 
			{
				case 0: 
				{
					if (j == 0)//beginning col
					{
						if (magArray[i][j] <= magArray[i][j + 1])
						{
							magGradOut[i][j] = 0;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else if (j == W - 1)
					{
						if (magArray[i][j] <= magArray[i][j - 1])
						{
							magGradOut[i][j] = 0;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else
					{
						if (myMax(magArray[i][j], magArray[i][j - 1], magArray[i][j + 1]) != magArray[i][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					break;
				}

				case 1:
				{
					if ((j == 0 && i == 0) || (j == W - 1 && i == W - 1))
					{
						magGradOut[i][j] = magArray[i][j];
					}
					else if ((j == 0 && i != 0) || (j != W - 1 && i == W - 1))//beginning col
					{
						if (magArray[i][j] <= magArray[i - 1][j + 1])
						{
							magGradOut[i][j] = 0;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else if ((j != 0 && i == 0) || (j == W - 1 && i != W - 1))
					{
						if (magArray[i][j] <= magArray[i + 1][j - 1])
						{
							magGradOut[i][j] = 0;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else
					{
						if (myMax(magArray[i][j], magArray[i + 1][j - 1], magArray[i - 1][j + 1]) != magArray[i][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					break;
				}

				case 2:
				{
					if (i == 0)//beginning col
					{
						if (magArray[i][j] <= magArray[i + 1][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else if (i == W - 1)
					{
						if (magArray[i][j] <= magArray[i-1][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else
					{
						if (myMax(magArray[i][j], magArray[i-1][j], magArray[i+1][j]) != magArray[i][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					break;
				}

				case 3:
				{
					if ((j == W - 1 && i == 0) || (j == 0 && i == W - 1))
					{
						magGradOut[i][j] = magArray[i][j];
					}
					else if ((j == 0 && i != W - 1) || (j != W - 1 && i == 0))//beginning col
					{
						if (magArray[i][j] <= magArray[i + 1][j + 1])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else if ((j != 0 && i == W - 1) || (j == W - 1 && i != 0))
					{
						if (magArray[i][j] <= magArray[i - 1][j - 1])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					else
					{
						if (myMax(magArray[i][j], magArray[i + 1][j + 1], magArray[i - 1][j - 1]) != magArray[i][j])
						{
							magGradOut[i][j] = 0;
							removed++;
						}
						else
						{
							magGradOut[i][j] = magArray[i][j];
						}
					}
					break;
				}
				default:
					magGradOut[i][j]=0;
					cout << "sth wrong: " << magArray[i][j] << endl;
					break;
			}
		}
	}
    //print those pixel I got rid of using the histogram thresholding
	// cout << "number of nms point: "<<removed << " out of "<<H*W<<endl;
	return 1;
}

bool histoBuild(int(&array)[H][W])
{	
	int rowArray = H;
	int colArray = W;
	for (int i = 0; i < rowArray; i++)
	{
		for (int j = 0; j < colArray; j++)
		{
			if (array[i][j] <= 255 && array[i][j] >= 0)
			{
				countTable[array[i][j]]++;
			}
		}
	}
	return 1;
}

// Function for thresholding
bool thresHolding(int(&array)[H][W],bool isNaive, int threshold)
{	
	int rowArray = H;
	int colArray = W;
	
    //naive thresholding, only take pixel with value larger than certain number
	if (isNaive == 1)
	{
		for (int i = 0; i < rowArray; i++)
			for (int j = 0; j < colArray; j++)
			{
				if (array[i][j] <= threshold)
				{
					array[i][j] = 0;
				}
                else{
                    array[i][j] = 255;
                }
                
               
			}
	}
    //Here we use ptile method
	else if (isNaive == 0)
	{
		int fin_thres = 0;
		int tmp_sum = 0;
        int nzpNum = 0;
		
        // cout << threshold << "% " <<"number of above threshold: "<< int(threshold / 100.0*rowArray*colArray) <<"/"<<rowArray*colArray<< ' ' <<endl;
		//cout << "index 0: "<<countTable[0] << endl;
        
        //find which greyscale value we should take as the threshold
        //based on the percantage we provided 50% 30% 10%
        //countTable for storing histogram num
        // add up from 1 to 255 to get number of non zero pixels
        for (int i = 1; i <= 255; i++)
        {
            nzpNum += countTable[i];
        }
		for (int i = 0; i < 255; i++)
		{
			tmp_sum += countTable[255 - i];
			//cout << tmp_sum << endl;
			if (tmp_sum >= int(threshold/100.0*nzpNum))
			{
				fin_thres = 255 - i;
				break;
			}
		}
		// cout << "threshold: " << fin_thres << endl;
		thresHolding(array, 1, fin_thres);
	}
	return  1;
}

// auxillary function for showing the max of these three values
int myMax(int a, int b, int c)
{
	if (a >= b)
	{
		if (a >= c)
			return a;
		else
			return c;
	}
	else
	{
		if (b >= c)
			return b;
		else
			return c;
	}
}

//auxillary function for transforming 2d array to 1d
bool fill(int(&array1)[H][W], int(&array2)[H][W],int div,int reduce)
{
	for (int i = reduce;i < H-reduce; i++)
	{
		for (int j = reduce; j < W-reduce; j++)
		{
			array2[i][j] = abs(array1[i][j])/div;
			if(array2[i][j] > 255)
			{
				array2[i][j] = 0;
			}
		}
	}
	
	return 1;
}

bool fill_opt(int(&array1)[H][W], int(&array2)[H][W],int div,int reduce)
{
	#pragma omp parallel for collapse(2) num_threads(4)
	for (int i = reduce;i < H-reduce; i++)
	{
		for (int j = reduce; j < W-reduce; j++)
		{
			array2[i][j] = abs(array1[i][j])/div;
			if(array2[i][j] > 255)
			{
				array2[i][j] = 0;
			}
		}
	}
	
	return 1;
}
