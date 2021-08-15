'''
Script for converting Eigen vector/matrix output to arrays that javascript can use.
'''
import re

def read_bias(infile, outfile):
    outfile.write("[")
    first = True
    line = infile.readline()
    while line and not line.isspace():
        if first:
            first = False
        else:
            outfile.write(",")
        outfile.write(line.strip())
        line = infile.readline()
    outfile.write("]")

def read_weight(infile, outfile):
    outfile.write("[\n")

    first = True
    line = infile.readline()
    while line and not line.isspace():
        if first:
            first = False
        else:
            outfile.write(",\n")
        outfile.write("[")
        outfile.write(re.sub("\s+", ",", line.strip()))
        outfile.write("]")
        line = infile.readline()

    outfile.write("\n]")

filename = "weights_and_biases.txt"

num_layers = 4

f = open(filename, 'r')
biases = []
weights = []
for i in range(1, num_layers):
    biases.append(open("b" + str(i) + ".txt", 'w'))
    weights.append(open("w" + str(i) + ".txt", 'w'))

for i in range(1, num_layers):
    read_bias(f, biases[i - 1])
    read_weight(f, weights[i - 1])

f.close();
for i in range(1, num_layers):
    biases[i-1].close()
    weights[i-1].close()
