//  util.h
//  cannyEdgeDectector
#pragma once
#ifndef UTIL_H
#define UTIL_H

#include"string.h"
#include <iostream>
#include <fstream>

//!! size of the image, ready to be changed by bash
//lena: 512*512 24
//Lena256: 256*256  8
//zebra: 443*665(pad+3) 8
//tiger: 354*630(pad+2) 24

// #define H  3000
// #define W 4006
#define H 3000
#define W 4006

extern int num_Chn;
extern int isRead;
extern int org_img[H][W];
extern int img[H][W];

bool readImg(std::string);
bool writeImg(std::string,int(&array)[H][W]);
bool iniArray(size_t*);

#endif // !UTIL.H
