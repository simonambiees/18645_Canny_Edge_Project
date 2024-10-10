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

#define num_of_run 100

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

    tsc_counter t0, t1;
    long long sum = 0;
    double average_cycle = 0;

    cout << "start processing" << endl;
    if (readImg(pwdIn+nameIn))
    {
        cout << "opened file complete!" << endl;
        // to print a sample 10*10 subimage to see if read correctly
        // for (int i = 0; i < 10; i++)
        // {
        //     for (int j = 0; j < 10; j++)
        //         cout << img[i][j] << " ";
        //     cout << endl;
        // }
    }
    else
    {
        cout << "opened file failed" << endl;
        return 0;
    }

    for(int i = 0; i < num_of_run; i++){
        initial();
        RDTSC(t0);
        gausFilter(img);
        
        //Function: gradient magnitude and direction computing
        gradientForm(img,1);
        nms(magGrad, dirGrad);//nms
        histoBuild(magGradOut);//build histogram array.
        fill(magGradOut,magGradOut3,1,5);
        thresHolding(magGradOut3, 0, 30);//threshold 0.3
        RDTSC(t1);
        sum += (long long)COUNTER_DIFF(t1, t0, CYCLES);
        //printf("Count time: %llu\n", (long long)COUNTER_DIFF(t1, t0, CYCLES));
        //printf("sum: %llu\n", sum);
    }
    average_cycle = (double) (sum / ((double) num_of_run));
    printf("Raw Performance Average Count time: %lf cycles\n\n", average_cycle);
    
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
        memcpy(img, org_img, sizeof(img));
}