// Copyright @2018 xuxiang. All rights reserved.

#include "nn.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace nn;

int main(int argc, char** argv) {
  //cout << "topology: 2 4 1" << endl;
	//for(int i = 2000; i >= 0; --i) {
		//double a = (rand() / double(RAND_MAX));
		//double b = (rand() / double(RAND_MAX));
		//int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		//int n2 = (int)(2.0 * rand() / double(RAND_MAX));
    //cout << a << " " << b << endl;
		//int t = n1 ^ n2; // should be 0 or 1
		//cout << t << ".0" << endl; 
	//}

  FCN fcn("/home/xuxiang/work/dl-tutorial/nn-from-scratch/train_set.in");
  cout << "end" << endl;
  return 0;
}
