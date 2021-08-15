# see https://stackoverflow.com/questions/36659004/eigen-matrix-multiplication-speed
g++-10 train_network.cpp network.cpp read_mnist.cpp -Iinclude -Ofast -fopenmp 
#g++-10 train_network.cpp network.cpp read_mnist.cpp -Iinclude -O3 -DNDEBUG -fopenmp
#g++-10 -framework Accelerate -DEIGEN_USE_BLAS -O3 -DNDEBUG train_network.cpp network.cpp read_mnist.cpp -Iinclude
