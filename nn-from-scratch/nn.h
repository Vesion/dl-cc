// Copyright @2018 xuxiang. All rights reserved.

#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

namespace nn {

//
// =============================Fully Connected Network=============================
//
class FCN {
public:
  FCN() = delete;
  FCN(const string& filename);
  ~FCN();
  void StartTraining(unsigned epochs, double learning_rate);
  double GetResult() const;
  double GetLossError() const;

  using Matrix = vector<vector<double>>;
  using Layer = vector<double>;

private:
  static double ActiveFunction(double x) { return tanh(x); }
  static double ActiveFuntionDerivative(double x) { return 1.0f - x*x; }
  static double LossFunction(double yo, double yt) { return 0.5 * (yo-yt) * (yo-yt); }
  static double GetRandom() { return rand() / double(RAND_MAX); }

  void GetNextInputs();
  void IterateOneEpoch();
  void FeedForward();
  void Backpropagation();

  void PrintWeights() {
    for (unsigned i = 0; i < weights_.size(); ++i) {
      auto& x = weights_[i];
      unsigned rs = x.size(), cs = x[0].size();
      cout << "matrix[" << rs << "," << cs << "]" << endl;
      for (unsigned r = 0; r < rs; ++r) {
        for (unsigned c = 0; c < cs; ++c) {
          cout << x[r][c] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
  }

  // output of each layer:
  // net_[0] is input layer, net_[1:-1] are hidden layers, net_[-1] is output layer
  vector<Layer> net_;
  unsigned num_layers_;
  double label_;
  vector<Matrix> weights_;

  double eta; // learning rate

  ifstream fin_;
  streampos data_start_p_;
};

FCN::FCN(const string& filename) {
  fin_.open(filename.c_str(), ios::in);
  string line, label;
  getline(fin_, line);
  stringstream ss(line);
  ss >> label;
  if (label != "topology:") {
    cerr << "Invalid training data file. " << endl;
    abort();
  }
  unsigned d;
  vector<unsigned> nums;
  while (ss >> d) nums.emplace_back(d);
  num_layers_ = nums.size();

  net_.resize(num_layers_);
  weights_.resize(num_layers_-1);
  for (unsigned i = 0; i < nums.size(); ++i) {
    net_[i].resize(nums[i]); // init outputs
    if (i != nums.size()-1) { // init weights
      weights_[i] = vector<vector<double>>(nums[i], vector<double>(nums[i+1]));
      for (unsigned r = 0; r < weights_[i].size(); ++r) {
        for (unsigned c = 0; c < weights_[i][r].size(); ++c) {
          weights_[i][r][c] = GetRandom();
        }
      }
    }
  }
  PrintWeights();
  data_start_p_ = fin_.tellg();
}

FCN::~FCN() {
  if (fin_.is_open()) fin_.close();
}

double FCN::GetResult() const {
  return net_.back()[0]; // the output layer
}

double FCN::GetLossError() const {
  return LossFunction(GetResult(), label_);
}

void FCN::GetNextInputs() {
  string line;
  getline(fin_, line);
  stringstream ss(line); 
  unsigned numFeatures = net_[0].size();
  for (unsigned i = 0; i < numFeatures; ++i) {
    ss >> net_[0][i];
  }
  getline(fin_, line);
  ss.str(line);
  ss >> label_;
}

void FCN::StartTraining(unsigned epochs, double learning_rate) {
  eta = learning_rate;
  for (unsigned epoch = 1; epoch <= epochs; ++epoch) {
    if (fin_.eof()) fin_.seekg(data_start_p_);
    IterateOneEpoch();
  }
}

void FCN::IterateOneEpoch() {
  while (!fin_.eof()) {
    GetNextInputs();
    FeedForward();
    cout << "Loss Error: " << GetLossError() << endl;
    Backpropagation();
  }
}

void FCN::FeedForward() {
  for (unsigned layer = 1; layer < num_layers_; ++layer) {
    int i = layer-1, j = layer;
    for (unsigned a = 0; a < net_[j].size(); ++a) {
      double sum = 0.0;
      for (unsigned b = 0; b < net_[i].size(); ++b) {
        sum += net_[i][b] * weights_[i][b][a];
      }
      net_[j][a] = sum;
    }
  }
}

void FCN::Backpropagation() {
  //double result = GetResult();
  // eta
}

} // end namespace
