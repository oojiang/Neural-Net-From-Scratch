#include <network.hpp>
#include <read_mnist.hpp>
#include <stdlib.h> /* atoi */

/* Trains a neural net for digit classification using the MNIST dataset, and
 * saves the network to a file.
 * A lot of things in this program are hard coded (bad), but I couldn't
 * be bothered to fix it up. */
int main(int argc, char *argv[]) {
  vector<int> sizes;
  sizes.push_back(784);
  sizes.push_back(200);
  sizes.push_back(100);
  sizes.push_back(10);

  vector<int> training_labels = get_labels("data/train-labels-idx1-ubyte");
  vector<VectorXd> training_inputs = get_images("data/train-images-idx3-ubyte");
  vector<VectorXd> training_outputs = get_output_vectors(training_labels);

  vector<int> test_labels = get_labels("data/t10k-labels-idx1-ubyte");
  vector<VectorXd> test_inputs = get_images("data/t10k-images-idx3-ubyte");

  int num_epochs = 100;
  if (argc > 1) {
    num_epochs = atoi(argv[1]);
  }
  int mini_batch_size = 100;
  double learning_rate = 0.1;
  cout << "batch size: " << mini_batch_size << endl;
  cout << "learning rate: " << learning_rate << endl;
  cout << "number of epochs: " << num_epochs << endl;
  Network n(sizes, training_inputs, training_outputs, test_inputs, test_labels, mini_batch_size, learning_rate);
  n.evaluate();
  cout << "Begin SGD" << endl;
  n.stochastic_gradient_descent(num_epochs);
  cout << "Finish SGD" << endl;
  int success = n.save_weights_and_biases("weights_and_biases.txt");
  if (success == 0) {
    cout << "Successfully saved." << endl;
  } else {
    cout << "Save failed." << endl;
  }
  return 0;
}
