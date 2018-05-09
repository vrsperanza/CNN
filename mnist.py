import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import numpy as np
def parse_idx(fd):
	DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)
	
	header = fd.read(4)

	zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
	data_type = DATA_TYPES[data_type]
	dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))

	data = array.array(data_type, fd.read())
	data.byteswap()

	expected_items = functools.reduce(operator.mul, dimension_sizes)

	return np.array(data).reshape(dimension_sizes)
	
def trainingImages():
	return parse_idx(open('train-images.idx3-ubyte', 'rb'))
def trainingLabels():
	return parse_idx(open('train-labels.idx1-ubyte', 'rb'))
def testingImages():
	return parse_idx(open('t10k-images.idx3-ubyte', 'rb'))
def testingLabels():
	return parse_idx(open('t10k-labels.idx1-ubyte', 'rb'))