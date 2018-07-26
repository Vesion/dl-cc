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
//    w[i][j] = w[i][j] - learning_rate * d(E)/d(w[i][j])
//
//  Backpropagation:
//    d(E)/d(yo) = yo - yt, L2 loss function
//    d(E)/d(x[j]) = d(E)/d(y[j]) * d(y[j])/d(x[j])
//                 = d(E)/d(y[j]) * f'(x[j])
//    d(E)/d(w[i][j]) = d(E)/d(x[j]) * d(x[j])/d(w[i][j])
//                    = d(E)/d(x[j]) * y[i]
//    d(E)/d(y[i]) = ∑( d(E)/d(x[j]) * d(x[j])/d(y[i]) ), j ∈ out(i)
//                 = ∑( d(E)/d(x[j]) * w[i][j] ), j ∈ out(i)
//

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

namespace nn {

class FullyConnectedNetwork {
public:
  FullyConnectedNetwork() = delete;
  FullyConnectedNetwork(const string& filename);
  ~FullyConnectedNetwork();
  void StartTraining(unsigned epochs, double learning_rate);
  double GetOutput() const;
  double GetLossError() const;

  using Matrix = vector<vector<double>>;
  using Layer = vector<double>;

private:
  static double ActiveFunction(double x) { return tanh(x); }
  static double ActiveFuntionDerivative(double x) { return 1.0f - x*x; }
  static double LossFunction(double yo, double yt) { return 0.5 * (yo-yt) * (yo-yt); }
  static double LossFunctionDerivative(double yo, double yt) { return yo - yt; }
  static double GetRandom() { return rand() / double(RAND_MAX); }

  void IterateOneEpoch();
  bool GetNextInputs();
  void FeedForward();
  void Backpropagation();

  void _PrintMatrix(const vector<vector<double>>& m) {
    for (unsigned i = 0; i < m.size(); ++i) {
      for (unsigned j = 0; j < m[i].size(); ++j) {
        cout << m[i][j] << " ";
      }
      cout << endl;
    }
  }
  void _PrintXs() {
    cout << "X: " << endl;
    _PrintMatrix(x_);
  }
  void _PrintYs() {
    cout << "Y: " << endl;
    _PrintMatrix(y_);
  }
  void _PrintWeights() {
    for (unsigned i = 0; i < weights_.size(); ++i) {
      cout << "Weights " << i << ": " << endl;
      _PrintMatrix(weights_[i]);
    }
  }

  unsigned num_layers_;
  vector<Layer> x_, y_; // include input layer, hidden layers, output layer
  vector<Matrix> weights_;

  double label_; // label of each input example
  double output_; // output of each input example
  double eta; // learning rate

  ifstream fin_;
  streampos data_start_p_;
};

FullyConnectedNetwork::FullyConnectedNetwork(const string& filename) {
  fin_.open(filename.c_str(), ios::in);
  if (!fin_.is_open()) {
    cerr << "No train set input. " << endl;
    abort();
  }

  // read topology, each number for a size of a layer
  string line, t;
  getline(fin_, line);
  stringstream ss(line);
  ss >> t;
  if (t != "topology:") {
    cerr << "Need a topology to initialize network. " << endl;
    abort();
  }
  unsigned d;
  vector<unsigned> nums;
  while (ss >> d) nums.emplace_back(d);
  num_layers_ = nums.size();

  // initialize x, y and weights
  x_.resize(num_layers_);
  weights_.resize(num_layers_-1);
  for (unsigned i = 0; i < nums.size(); ++i) {
    x_[i].resize(nums[i]);
    if (i != nums.size()-1) { // init weights
      weights_[i] = vector<vector<double>>(nums[i], vector<double>(nums[i+1]));
      for (unsigned r = 0; r < weights_[i].size(); ++r) {
        for (unsigned c = 0; c < weights_[i][r].size(); ++c) {
          weights_[i][r][c] = GetRandom();
        }
      }
    }
  }
  y_ = x_;

  // store the position of the beginning
  data_start_p_ = fin_.tellg();
}

FullyConnectedNetwork::~FullyConnectedNetwork() {
  if (fin_.is_open()) fin_.close();
}

double FullyConnectedNetwork::GetOutput() const {
  return output_;
}

double FullyConnectedNetwork::GetLossError() const {
  return LossFunction(output_, label_);
}

void FullyConnectedNetwork::StartTraining(unsigned epochs, double learning_rate) {
  eta = learning_rate;
  for (unsigned epoch = 1; epoch <= epochs; ++epoch) {
    fin_.seekg(data_start_p_);
    IterateOneEpoch();
  }
}

void FullyConnectedNetwork::IterateOneEpoch() {
#ifdef __NN_DEBUG
  int example_id = 1;
#endif
  while (!fin_.eof()) { // for each input example, do forward, backpropa and calc loss
    if (GetNextInputs()) {
#ifdef __NN_DEBUG
      cout << "No. " << example_id++ << endl;
      cout << "Before iteration:" << endl;
      _PrintXs();
      _PrintYs();
      _PrintWeights();
#endif

      FeedForward();
      Backpropagation();
      
#ifdef __NN_DEBUG
      cout << endl << "After iteration:" << endl;
      _PrintXs();
      _PrintYs();
      _PrintWeights();
      cout << "Loss Error: " << GetLossError() << endl;
      cout << endl;
#endif
    }
  }
}

bool FullyConnectedNetwork::GetNextInputs() {
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

// calculate x and y of each neuron
void FullyConnectedNetwork::FeedForward() {
  y_[0] = x_[0];
  for (unsigned l = 1; l < num_layers_; ++l) {
    for (unsigned j = 0; j < x_[l].size(); ++j) {
      double sum = 0.0;
      for (unsigned i = 0; i < x_[l-1].size(); ++i) {
        sum += x_[l-1][i] * weights_[l-1][i][j]; // TODO: add bias here
      }
      x_[l][j] = sum;
      y_[l][j] = ActiveFunction(sum);
    }
  }
  output_ = y_.back()[0];
}

void FullyConnectedNetwork::Backpropagation() {
  auto dEy = x_, dEx = x_;

  for (int l = num_layers_-1; l > 0; --l) {

    if (l == (int)num_layers_-1) { // the output layer

      // d(E)/d(yo) = yo - yt
      dEy[l][0] = LossFunctionDerivative(output_, label_);

      // d(E)/d(x) = d(y)/d(x) * d(E)/d(y)
      dEx[l][0] = ActiveFuntionDerivative(output_) * dEy[l][0];

    } else { // hidden layer

      // d(E)/d(x) = d(y)/d(x) * d(E)/d(y)
      for (unsigned j = 0; j < dEx[l].size(); ++j) {
        dEx[l][j] = ActiveFuntionDerivative(y_[l][j]) * dEy[l][j];
      }

      // d(E)/d(y[i]) = ∑( d(E)/d(x[j]) * d(x[j])/d(y[i]) ), j ∈ out(i)
      //              = ∑( d(E)/d(x[j]) * w[i][j] ), j ∈ out(i)
      for (unsigned i = 0; i < x_[l-1].size(); ++i) {
        double sum = 0.0;
        for (unsigned j = 0; j < x_[l].size(); ++j) {
          sum += dEx[l][j] * weights_[l-1][i][j];
        }
        dEy[l-1][i] = sum;
      }
    }

    // d(E)/d(w[i][j]) = d(E)/d(x[j]) * d(x[j])/d(w[i][j]) 
    //                 = d(E)/d(x[j]) * y[i]
    auto& weight = weights_[l-1];
    for (unsigned i = 0; i < weight.size(); ++i) {
      for (unsigned j = 0; j < weight[0].size(); ++j) {
        double dEw = y_[l-1][i] * dEx[l][j]; // gradient
        weight[i][j] -= eta * dEw; // graident descent
      }
    }
  }
}
} // end namespace

