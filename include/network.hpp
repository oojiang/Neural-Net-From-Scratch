#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Network {
  public:
    Network(vector<int> &sizes, vector<VectorXd> &training_inputs, vector<VectorXd> &training_outputs,
       vector<VectorXd> &test_inputs, vector<int> &test_labels,
       int mini_batch_size, double learning_rate);
    void feedforward(VectorXd &a);
    void backpropagate(VectorXd &y);
    void stochastic_gradient_descent(int epochs);
    double evaluate();
    int save_weights_and_biases(string filepath);
  private:
    void update_mini_batch(vector<int> &indices, int batch_num); /* helper for ``stochastic_gradient_descent``. */
    VectorXd sigmoid_prime(int &layer_num); /* helper for ``backpropogate``. */
    VectorXd relu_prime(int &layer_num); /* helper for ``backpropogate``. */
    VectorXd cost_derivative(VectorXd &y); /* helper for ``backpropogate``. */
    int num_layers_; /* Number of layers in the network (including input and output layers). */
    vector<int> sizes_; /* Vector containing the sizes of each layer. ``sizes_.size() == num_layers``. */
    vector<VectorXd> biases_; /* Bias vectors for each layer. 
                               * 0th index contains a dummy vector since there is no bias for the input layer. */
    vector<MatrixXd> weights_; /* Weight vectors for each layer except the input. 
                                * 0th index contains a dummy matrix since there is no weight for the input layer. */
    vector<VectorXd> as_; /* Stores the activations of each layer to avoid repeated computation. */
    vector<VectorXd> zs_; /* Stores the z-vectors of each layer to avoid repeated computation. */
    vector<VectorXd> nabla_b_; /* Stores the derivatives of the cost function with respect to the baises. */
    vector<MatrixXd> nabla_w_; /* Stores the derivatives of the cost function with respect to the weights. */
    vector<VectorXd> training_inputs_;
    vector<VectorXd> training_outputs_;
    vector<VectorXd> test_inputs_;
    vector<int> test_labels_;
    int mini_batch_size_;
    double learning_rate_;
};

