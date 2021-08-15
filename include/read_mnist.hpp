#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void print_image(VectorXd image, int num_rows, int num_cols);
void print_image_values(VectorXd image, int num_rows, int num_cols);
vector<int> get_labels(string path);
vector<VectorXd> get_images(string path);
vector<VectorXd> get_output_vectors(vector<int> labels);
