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

#define num_of_run 10

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

    // tsc_counter t0, t1;
    long long sum_normal = 0;
    long long sum_SIMD = 0;
    double average_cycle_normal = 0;
    double average_cycle_SIMD = 0;
    double sum_tp = 0.0;
    double average_tp = 0.0;

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
        // initial();
        // // normal run
        // RDTSC(t0);
        // gausFilter(img);
        // gradientForm(img,1);
        // nms(magGrad, dirGrad);
        // histoBuild(magGradOut);
        // fill(magGradOut,magGradOut3,1,5);
        // thresHolding(magGradOut3, 0, 30);
        // RDTSC(t1);
        // sum_normal += (long long)COUNTER_DIFF(t1, t0, CYCLES);
        // // SIMD run
        // initial();
        // RDTSC(t0);
        // gausFilter_SIMD(img);
        // gradientForm_SIMD(img,1);
        // nms(magGrad, dirGrad);
        // histoBuild(magGradOut);
        // fill(magGradOut,magGradOut3,1,5);
        // thresHolding(magGradOut3, 0, 30);
        // RDTSC(t1);
        // sum_SIMD += (long long)COUNTER_DIFF(t1, t0, CYCLES);

        initial();
        sum_tp += convol_kernel_bench_wrapper(img);

    }
    average_cycle_normal = (double) (sum_normal / ((double) num_of_run));
    average_cycle_SIMD = (double) (sum_SIMD / ((double) num_of_run));
    average_tp = sum_tp / ((double)num_of_run);
    printf("\nAverage Throughput: %lf\nOver %d runs\n", average_tp, num_of_run);
    // printf("\n%d runs\n\n", num_of_run);
    // printf("Normal Raw Performance Average Count time: %lf cycles\n\n", average_cycle_normal);
    // printf("SIMD Raw Performance Average Count time: %lf cycles\n\n", average_cycle_SIMD);
    
    cout<<"press any key to end:"<<endl;
    cin >> ch;
}

//This function initializes all variables
void initial(void){
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