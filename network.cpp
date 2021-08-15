#include <network.hpp>
#include <ctime> /* clock(), CLOCKS_PER_SEC */
#include <fstream> /* ofstream */
#include <string.h> /* to_string */

inline VectorXd NULL_V() { return VectorXd::Constant(0, 0); }
inline MatrixXd NULL_M() { return MatrixXd::Constant(0, 0, 0); }
inline double sigmoid(double x) { return 1 / ( 1 + exp(-x)); }
inline VectorXd sigmoidv(VectorXd v) { return v.unaryExpr(std::ref(sigmoid)); }
inline double sigmoid_prime_helper(double x) { return x * (1 - x); }

/* Constructor initializes ``biases_`` and ``weights__`` with random values 
   between -1 and 1. */
Network::Network(vector<int> &sizes, vector<VectorXd> &training_inputs, vector<VectorXd> &training_outputs, 
    vector<VectorXd> &test_inputs, vector<int> &test_labels,
    int mini_batch_size, double learning_rate) {
  num_layers_ = sizes.size();
  sizes_ = sizes;
  biases_.push_back(NULL_V());
  weights_.push_back(NULL_M());
  for (int i=1; i < num_layers_; i++) {
    biases_.push_back(VectorXd::Random(sizes_[i]));
    weights_.push_back(MatrixXd::Random(sizes_[i], sizes_[i-1]));
  }

  training_inputs_ = training_inputs;
  training_outputs_ = training_outputs;
  test_inputs_ = test_inputs;
  test_labels_ = test_labels;
  mini_batch_size_ = mini_batch_size;
  learning_rate_ = learning_rate;
}

/* Computes the output of the network given input ``a0``. */
void Network::feedforward(VectorXd &a0) {
  VectorXd activation = a0;
  zs_.clear();
  zs_.push_back(NULL_V());
  as_.clear();
  as_.push_back(a0);
  for (int i=1; i < num_layers_; i++) {
    activation = weights_[i] * activation + biases_[i];
    zs_.push_back(activation);
    activation = sigmoidv(activation);
    as_.push_back(activation);
  }
}

/* Computes ``nabla_b_`` and ``nabla_w_``. 
 * Assumes that feedforward has already been called. */
void Network::backpropagate(VectorXd &y) {
  nabla_b_.clear();
  nabla_w_.clear();
  for (int i = 0; i < num_layers_; i++) {
    nabla_b_.push_back(NULL_V());
    nabla_w_.push_back(NULL_M());
  }

  int curr_layer = num_layers_ - 1;
  VectorXd delta = cost_derivative(y).cwiseProduct(sigmoid_prime(curr_layer));
  
  nabla_b_[curr_layer] = delta;
  nabla_w_[curr_layer] = delta * as_[curr_layer - 1].transpose();

  for (curr_layer--; curr_layer > 0; curr_layer--) {
    delta = (weights_[curr_layer + 1].transpose() * delta).cwiseProduct(sigmoid_prime(curr_layer));
    nabla_b_[curr_layer] = delta;
    nabla_w_[curr_layer] = delta * as_[curr_layer-1].transpose();
  }
}

/* Performs stochastic gradient descent. Evaluates the performance of the NN
 * on the training set and the test set every 10 epochs. Saves the weights
 * and biases to a file at the end. I also hard coded some accuracy milestones
 * for this function to save the weights and biases at, in case training 
 * gets terminated early */
void Network::stochastic_gradient_descent(int epochs) {
  double milestones[] = { 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1};
  int mi = 0;
  cout << "Beginning stochastic gradient descent" << endl;
  clock_t sgd_time = clock();
  vector<int> indices;
  for (int i = 0; i < training_inputs_.size(); i++) {
    indices.push_back(i);
  }
  std::random_shuffle(indices.begin(), indices.end());

  clock_t epoch_time;
  double accuracy;
  for (int e = 0; e < epochs; e++) {
    cout << endl << endl << "Beginning Epoch " << e << "." << endl;
    epoch_time = clock();
    for (int batch_num = 0; batch_num * mini_batch_size_ < training_inputs_.size(); batch_num++) {
      update_mini_batch(indices, batch_num);
    }
    cout << "Finished Epoch " << e << ". Took " << (float)(clock() - epoch_time) / CLOCKS_PER_SEC << "seconds." << endl;
    cout << endl;
    if (e % 10 == 0) {
      accuracy = evaluate();
      if (accuracy > milestones[mi]) {
        save_weights_and_biases("3wnb_" + to_string(accuracy) + ".txt");
        while (accuracy > milestones[mi]) {
          mi++;
        }
      }
    }
  }
  evaluate();
  cout << "Finished stochastic gradient descent. Took " << (float)(clock() - sgd_time) / CLOCKS_PER_SEC << "seconds." << endl;
}

/* ``indices`` contains the indices in the minibatch. */
void Network::update_mini_batch(vector<int> &indices, int batch_num) {
  int index;
  VectorXd a0, y;
  vector<VectorXd> nabla_b_sum;
  vector<MatrixXd> nabla_w_sum;
  for (int b = batch_num * mini_batch_size_; 
      b < (batch_num + 1) * mini_batch_size_ && b < training_outputs_.size(); 
      b++) {
    index = indices[b];
    a0 = training_inputs_[index];
    y = training_outputs_[index];
    feedforward(a0);
    backpropagate(y);
    if (nabla_b_sum.size() == 0) {
      for (int i = 0; i < num_layers_; i++) {
        nabla_b_sum.push_back(nabla_b_[i]);
        nabla_w_sum.push_back(nabla_w_[i]);
      }
    } else {
      for (int i = 0; i < num_layers_; i++) {
        nabla_b_sum[i] += nabla_b_[i];
        nabla_w_sum[i] += nabla_w_[i];
      }
    }
  }

  for (int i = 0; i < num_layers_; i++) {
    biases_[i] -= learning_rate_ / mini_batch_size_ * nabla_b_sum[i];
    weights_[i] -= learning_rate_ / mini_batch_size_ * nabla_w_sum[i];
  }
}

/* Print out the accuracy on both the training and test sets, then
   return the accuracy on the test set. */
double Network::evaluate() {
  vector<int> train_counter;
  for (int i = 0; i < 10; i++) {
    train_counter.push_back(0);
  }
  double num_correct = 0;
  int output;
  int train_output;
  for (int i = 0; i < training_outputs_.size(); i++) {
    feedforward(training_inputs_[i]);
    as_.back().maxCoeff(&output);
    training_outputs_[i].maxCoeff(&train_output);
    train_counter[output]++;
    if (output == train_output) {
      num_correct++;
    }
  }

  cout << "Train accuracy: " << num_correct / training_inputs_.size() << endl;

  vector<int> test_counter;
  for (int i = 0; i < 10; i++) {
    test_counter.push_back(0);
  }
  num_correct = 0;
  for (int i = 0; i < test_labels_.size(); i++) {
    feedforward(test_inputs_[i]);
    as_.back().maxCoeff(&output);
    test_counter[output]++;
    if (output == test_labels_[i]) {
      num_correct++;
    }
  }

  cout << "Test accuracy: " << num_correct / test_labels_.size() << endl;

  return num_correct / test_labels_.size();
}

/* Prints the biases and weights to ``filepath`` and separates them
 * with an empty line. Prints the bias of the first layer, then the weight
 * of the first layer, then the bias of the second layer, and the weight
 * of the second layer, etc. */
int Network::save_weights_and_biases(string filepath) {
  std::ofstream file(filepath);
  if (file.is_open()) {
    for (int i = 1; i < num_layers_; i++) {
      file << biases_[i] << endl << endl;
      file << weights_[i] << endl << endl;
    }
    cout << "Successfully saved to " << filepath << "." << endl;
    return 0;
  }
  cout << "Failed to save to " << filepath << "." << endl;
  return 1;
}

/* Returns the derivative of the sigmoid of the z-vector of the ``layer_num``th layer. 
 * This function takes in a layer number instead of a vector because it allows us
 * to save some computation by using the saved activations. */
VectorXd Network::sigmoid_prime(int &layer_num) {
  return as_[layer_num].unaryExpr(std::ref(sigmoid_prime_helper));
}

/* Returns the derivative of the cost with respect to the output activations. */
VectorXd Network::cost_derivative(VectorXd &y) {
  return as_.back() - y;
}
