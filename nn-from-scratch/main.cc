// Copyright @2018 xuxiang. All rights reserved.

#include "nn.h"

#include <iostream>

using namespace std;
using namespace nn;

int main() {
  FullyConnectedNetwork fcn("train_set.in");
  fcn.StartTraining(1, 0.01);
  cout << "training end" << endl;
  return 0;
}
