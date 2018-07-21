// Copyright @2018 xuxiang. All rights reserved.
//
// Refer to:
// https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
//
//  Feedforward:
//    E = 1/2*(yo-yt)^2
//    y = f(x), f(x) = tanh(x), f'(x) = 1-f(x)*f(x)
//    x[j] = ∑( w[i][j]*y[i] + b[j] ), i ∈ in(j)
//
//  Update weights:
//    w[i][j] = w[i][j] - learning_rate*d(E)/d(w[i][j])
//
//  Backpropagation:
//    d(E)/d(yo) = yo - yt
//    d(E)/d(x) = f'(x) * d(E)/d(y)
//    d(E)/d(w[i][j]) = d(E)/d(x[j]) * d(x[j])/d(w[i][j]) = d(E)/d(x[j]) * y[i]
//    d(E)/d(y[i]) = ∑( d(E)/d(x[j]) * d(x[j])/d(y[i]) ), j ∈ out(i)
//

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

namespace nn {

class FCN { // Fully Connected Network
public:
  FCN() = delete;
  FCN(const string& filename);
  ~FCN();
  void StartTraining(unsigned epochs, double learning_rate);
  double GetOutput() const;
  double GetLossError() const;

  using Matrix = vector<vector<double>>;
  using Layer = vector<double>;

private:
  static double ActiveFunction(double x) { return tanh(x); }
  static double ActiveFuntionDerivative(double x) { return 1.0f - x*x; }
  static double LossFunction(double yo, double yt) { return 0.5 * (yo-yt) * (yo-yt); }
  static double GetRandom() { return rand() / double(RAND_MAX); }

  bool GetNextInputs();
  void IterateOneEpoch();
  void FeedForward();
  void Backpropagation();

  void _PrintWeights() {
    for (unsigned i = 0; i < weights_.size(); ++i) {
      auto& x = weights_[i];
      unsigned rs = x.size(), cs = x[0].size();
      cout << "weight[" << rs << "," << cs << "]" << endl;
      for (unsigned r = 0; r < rs; ++r) {
        for (unsigned c = 0; c < cs; ++c) {
          cout << x[r][c] << " ";
        }
        cout << endl;
      }
    }
  }

  void _PrintNet() {
    cout << "net:" << endl;
    for (unsigned i = 0; i < num_layers_; ++i) {
      for (unsigned j = 0; j < x_[i].size(); ++j) {
        cout << x_[i][j] << " ";
      }
      cout << endl;
    }
    cout << "output: " << output_ << endl;
  }

  unsigned num_layers_;
  vector<Layer> x_; // include input layer, hidden layers, output layer
  vector<Matrix> weights_;

  double label_; // label of each input example
  double output_; // output of each input example
  double eta; // learning rate

  ifstream fin_;
  streampos data_start_p_;
};

FCN::FCN(const string& filename) {
  fin_.open(filename.c_str(), ios::in);
  if (!fin_.is_open()) {
    cerr << "No train set input. " << endl;
    abort();
  }
  string line, label;
  getline(fin_, line);
  stringstream ss(line);
  ss >> label;
  if (label != "topology:") {
    cerr << "Need a topology to initialize fcn. " << endl;
    abort();
  }

  // read topology, each number for the size of a layer
  unsigned d;
  vector<unsigned> nums;
  while (ss >> d) nums.emplace_back(d);
  num_layers_ = nums.size();

  // initialize outputs and weights
  x_.resize(num_layers_);
  weights_.resize(num_layers_-1);
  for (unsigned i = 0; i < nums.size(); ++i) {
    x_[i].resize(nums[i]); // init outputs
    if (i != nums.size()-1) { // init weights
      weights_[i] = vector<vector<double>>(nums[i], vector<double>(nums[i+1]));
      for (unsigned r = 0; r < weights_[i].size(); ++r) {
        for (unsigned c = 0; c < weights_[i][r].size(); ++c) {
          weights_[i][r][c] = GetRandom();
        }
      }
    }
  }
  data_start_p_ = fin_.tellg(); // store the position of the beginning
}

FCN::~FCN() {
  if (fin_.is_open()) fin_.close();
}

double FCN::GetOutput() const {
  return output_;
}

double FCN::GetLossError() const {
  return LossFunction(output_, label_);
}

bool FCN::GetNextInputs() {
  string line;
  getline(fin_, line);
  if (line.empty()) return false;
  stringstream ss(line); 
  unsigned numFeatures = x_[0].size();
  for (unsigned i = 0; i < numFeatures; ++i) {
    ss >> x_[0][i];
  }
  getline(fin_, line);
  ss.str(line);
  ss >> label_;
  return true;
}

void FCN::StartTraining(unsigned epochs, double learning_rate) {
  eta = learning_rate;
  for (unsigned epoch = 1; epoch <= epochs; ++epoch) {
    fin_.seekg(data_start_p_);
    IterateOneEpoch();
  }
}

void FCN::IterateOneEpoch() {
  int i = 1;
  while (!fin_.eof()) { // for each input example, do forward, backprop, calc loss
    if (GetNextInputs()) {
      FeedForward();
      Backpropagation();
      // print
      cout << "line: " << i++ << endl;
      _PrintWeights();
      _PrintNet();
      cout << "Loss Error: " << GetLossError() << endl;
      cout << endl;
    }
  }
}

void FCN::FeedForward() {
  for (unsigned layer = 1; layer < num_layers_; ++layer) {
    int i = layer-1, j = layer;
    for (unsigned a = 0; a < x_[j].size(); ++a) {
      double sum = 0.0;
      for (unsigned b = 0; b < x_[i].size(); ++b) {
        sum += ActiveFunction(x_[i][b] * weights_[i][b][a]); // TODO: add bias here
      }
      x_[j][a] = sum;
    }
  }
  output_ = ActiveFunction(x_.back()[0]);
}

void FCN::Backpropagation() {
  //double output = GetOutput();
  // eta
}

} // end namespace
