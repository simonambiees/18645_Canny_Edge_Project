//
//  main.cpp
//  cannyEdgeDectector
//
//  Created by syd on 2018/10/31.
//  Copyright © 2018年 syd. All rights reserved.
//
#include <iostream>
#include <stdlib.h>
#include "main.h"
#include "rdtsc.h"

int num_of_run = 1;

double int_conversion_cycles_counter = 0.0;
using namespace std;
void initial(void);

int main(int argc, char* argv[])
{
    char ch;
    //picture address，ready to be changed by Bash Script
    string nameIn = argv[1];
    string pwd = "./";
    string pwdIn = pwd + "picIn/";
    string pwdOut = pwd+ "picOut/";
    if (argc < 2)
    {
        cout << "Please provide the image name" << endl;
        return 0;
    }
    if (argc == 3)
    {
        num_of_run = atoi(argv[2]);
    }

    tsc_counter t0, t1;
    double sum_convol_cycles_optimized = 0.0;
    double average_convol_cycles_optimized = 0.0;
    double sum_convol_cycles = 0.0;
    double average_convol_cycles = 0.0;
    double sum_canny_cycles_optimized = 0.0;
    double average_canny_cycles_optimized = 0.0;
    double sum_canny_cycles = 0.0;
    double average_canny_cycles = 0.0;
    double max_canny_speedup = 0.0;

    cout << "start processing" << endl;
    if (readImg(pwdIn+nameIn))
    {
        cout << "opened file complete!" << endl;
        // to print a sample 10*10 subimage to see if read correctly
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
                cout << img[i][j] << " ";
            cout << endl;
        }
    }
    else
    {
        cout << "opened file failed" << endl;
        return 0;
    }

    for(int i = 0; i < num_of_run; i++){
        initial();
        // normal canny run
        RDTSC(t0);
        gausFilter(img);
        gradientForm(img,1);
        nms(magGrad, dirGrad);
        histoBuild(magGradOut);
        fill(magGradOut,magGradOut3,1,5);
        thresHolding(magGradOut3, 0, 30);
        RDTSC(t1);
        double cur_canny_cycles = (long long)COUNTER_DIFF(t1, t0, CYCLES);
        sum_canny_cycles += cur_canny_cycles;
        // Optimized canny run
        initial();
        RDTSC(t0);
        gausFilter_opt(img);
        gradientForm_opt(img,1);
        nms(magGrad, dirGrad);
        histoBuild(magGradOut);
        fill_opt(magGradOut,magGradOut3,1,5);
        thresHolding(magGradOut3, 0, 30);
        RDTSC(t1);
        double cur_canny_cycles_optimized = ((long long)COUNTER_DIFF(t1, t0, CYCLES)-int_conversion_cycles_counter);
        sum_canny_cycles_optimized += cur_canny_cycles_optimized;

        if (cur_canny_cycles / cur_canny_cycles_optimized > max_canny_speedup)
        {
            max_canny_speedup = cur_canny_cycles / cur_canny_cycles_optimized;
        }

        // Isolating convolution kernel
        initial();
        sum_convol_cycles_optimized += convol_kernel_bench_wrapper(img);
        initial();
        sum_convol_cycles += convol_bench_wrapper(img);

    }
    average_convol_cycles = sum_convol_cycles / ((double)num_of_run);
    average_convol_cycles_optimized = sum_convol_cycles_optimized / ((double)num_of_run);
    average_canny_cycles = sum_canny_cycles / ((double)num_of_run);
    average_canny_cycles_optimized = sum_canny_cycles_optimized / ((double)num_of_run);
    printf("\nAverage CPU cycles (convol only)(Raw): %lf\nOver %d runs\n", average_convol_cycles, num_of_run);
    printf("\nAverage CPU cycles (convol only)(Optimzied): %lf\nOver %d runs\n", average_convol_cycles_optimized, num_of_run);
    printf("\nAverage CPU cycles (canny full)(Raw): %lf\nOver %d runs\n", average_canny_cycles, num_of_run);
    printf("\nAverage CPU cycles (canny full)(Optimzied): %lf\nOver %d runs\n", average_canny_cycles_optimized, num_of_run);
    printf("\nSpeedup (convol): %lf\n", average_convol_cycles / average_convol_cycles_optimized);
    printf("\nSpeedup (canny): %lf\n", average_canny_cycles / average_canny_cycles_optimized);
    printf("\nSpeedup (Max)(canny): %lf\n", max_canny_speedup);
    
    cout<<"press any key to end:"<<endl;
    cin >> ch;
}

//This function initializes all variables
void initial(void){
        int_conversion_cycles_counter = 0.0;
        memset(countTable, 0, sizeof(countTable));
        memset(oneDImgArray, 0, sizeof(oneDImgArray));
        memset(magGrad, 0, sizeof(magGrad));
        memset(magGradX, 0, sizeof(magGradX));
        memset(magGradY, 0, sizeof(magGradY));
        memset(dirGrad, 0, sizeof(dirGrad));
        memset(tmpConvArray, 0, sizeof(tmpConvArray));
        memset(magGradOut, 0, sizeof(magGradOut));
        memset(magGradOut3, 0, sizeof(magGradOut3));
        // memcpy(img, org_img, sizeof(img));
}