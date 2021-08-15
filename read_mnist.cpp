#include <read_mnist.hpp>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void print_image(VectorXd image, int num_rows, int num_cols) {
  int threshold = 25;
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      if (image[i * num_rows + j] > threshold) {
        cout << "*";
      } else {
        cout << ".";
      }
    }
    cout << endl;
  }
}

void print_image_values(VectorXd image, int num_rows, int num_cols) {
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      if (image[i * num_rows + j] == 0) {
        cout << "_._ ";
      } else {
        printf("%.1f ", ((float) image[i * num_rows + j]));
      }
    }
    cout << endl;
  }
}

vector<int> get_labels(string path) {
  ifstream file(path, ios::in|ios::binary);
  uint32_t magic;
  uint32_t num_items;
  char label;
  vector<int> labels;
  if (file.is_open()) {
    cout << "Reading label file: " << path << endl;

    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
      cout << "Incorrect magic number: " << magic << "(Should be 2049)" << endl;
      return labels;
    }

    file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);

    for (int item_id = 0; item_id < num_items; item_id++) {
      file.read(&label, 1);
      labels.push_back(label);
    }
  }
  return labels;
}

vector<VectorXd> get_images(string path) {
  ifstream file(path, ios::in|ios::binary);
  uint32_t magic;
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_cols;
  char* pixels;
  vector<VectorXd> images;

  if (file.is_open()) {
    cout << "Reading label file: " << path << endl;

    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
      cout << "Incorrect magic number: " << magic << "(Should be 2051)" << endl;
      return images;
    }

    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = swap_endian(num_images);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    num_rows = swap_endian(num_rows);
    file.read(reinterpret_cast<char*>(&num_cols), 4);
    num_cols = swap_endian(num_cols);

    cout << "Number of images: " << num_images << endl;
    cout << "Number of rows: " << num_rows << endl;
    cout << "Number of cols: " << num_cols << endl;

    pixels = new char[num_rows * num_cols];
    for (int i = 0; i < num_images; i++) {
      file.read(pixels, num_rows * num_cols);
      VectorXd image(num_rows * num_cols);
      for (int j = 0; j < num_rows * num_cols; j++) {
        if (pixels[j] < 0) {
          image[j] = (float)(((int) pixels[j]) + 255) / 255;
        } else {
          image[j] = (float)(pixels[j]) / 255;
        }
      }
      images.push_back(image);
    }
  }
  return images;
}

inline VectorXd output_vector(int label) { VectorXd y = VectorXd::Constant(10, 0); y[label] = 1; return y; }

vector<VectorXd> get_output_vectors(vector<int> labels) {
  vector<VectorXd> output_vectors;

  for (int i = 0; i < labels.size(); i++) {
    output_vectors.push_back(output_vector(labels[i]));
  }

  return output_vectors;
}
