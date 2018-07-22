// Copyright @2018 xuxiang. All rights reserved.

#include <iostream>
#include <fstream>

using namespace std;

int main() {
  ofstream fout("train_set.in", ios::trunc);
  fout << "topology: 2 4 1" << endl;
  for(int i = 0; i < 2000; ++i) {
    double a = (rand() / double(RAND_MAX));
    double b = (rand() / double(RAND_MAX));
    fout << a << " " << b << endl;
    int t = ((int)a) ^ ((int)b);
    fout << t << ".0" << endl; 
  }
  fout.close();
  return 0;
}
