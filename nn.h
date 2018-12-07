#ifndef NN_H
#define NN_H
/* Copyright (C) 2017 Lorne Schell All rights reserved. */
/* This is a concise sigmoid nn over mnist in c for illustration/education purposes.
   Note that it's single core and doesn't have some basic features such as minibatch*/
/* Results:
 * 88.550% testing accuracy with
  static const int hidden_neurons = 64, output_neurons = 10;
  static const float initialization_sigma = 0.50;
  static const int epochs = 20;
  float learning_rate = 0.01f, decay = 1.001f;
 */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <assert.h>
#include <errno.h>
#include <fcntl.h> // for read
#include <float.h> // for FLT_MIN
#include <math.h> // for all the math functions
#include <stdint.h> // for uint32_t's
#include <stdio.h> // for putchar, fprintf
#include <stdlib.h> // for rand
#include <string.h> // for strerror
#include <sys/mman.h> // for mmap
#include <sys/stat.h> // for open
#include <unistd.h> // for read

#include "err.h"

typedef struct matrix {
  size_t x, y;
  float *e;
} matrix;
struct neural_layer {
  matrix weights, w_delt, biases, b_delt, activations, zvals;
  float(*activate)(float zval, float bias);
  float(*backprop)(float weight);
};
struct dataset {
  uint32_t images, rows, columns;
  uchar *pixels;
  uchar *labels;
};

// swap the endianness, since this is on x86
retcode read_int(FD fp, uint32_t *r) {
  for (size_t i = 1; i <= sizeof(uint32_t); ++i) {
    retfail(read(fp, ((unsigned char *) r) + sizeof(uint32_t) - i, 1));
  }

  return SUCCESS;
}

// gunzip your mnist files beforehand - so we can mmap them here
retcode load_mnist(struct dataset *dat, char *image_file_name, char *label_file_name) {
  FD image_file = open(image_file_name, O_RDONLY);
  FD label_file = open(label_file_name, O_RDONLY);

  if (image_file == FAIL || label_file == FAIL) {
    fprintf(stderr, "%s:%d: Could not open the datafiles\n", __FILE__, __LINE__);
    perror(NULL);
    return FAIL;
  }

  uint32_t magic_num = 0;
  retfail(read_int(image_file, &magic_num));

  if (magic_num != 2051) {
    fprintf(stderr, "%s:%d: Image file magic num doesn't match %d\n", __FILE__, __LINE__, magic_num);
    return FAIL;
  }

  retfail(read_int(image_file, &dat->images));
  retfail(read_int(image_file, &dat->rows));
  retfail(read_int(image_file, &dat->columns));
  // for some reason, if I include the offset in the mmap call, it fails.
  dat->pixels = mmap(NULL, dat->images * dat->rows * dat->columns, PROT_READ, MAP_PRIVATE, image_file,
                     0) + sizeof(int32_t) * 4;

  if (dat->pixels == MAP_FAILED) {
    fprintf(stderr, "%s:%d: Map failed (%d): %s\n", __FILE__, __LINE__, errno, strerror(errno));
    return FAIL;
  }

  retfail(read_int(label_file, &magic_num));

  if (magic_num != 2049) {
    fprintf(stderr, "%s:%d: Image label magic num doesn't match %d\n", __FILE__, __LINE__, magic_num);
    return FAIL;
  }

  uint32_t label_count = 0;
  retfail(read_int(label_file, &label_count));

  if (dat->images != label_count) {
    fprintf(stderr, "%s:%d: Data count doesn't match labels %d and images %d\n", __FILE__, __LINE__, dat->images,
            label_count);
    return FAIL;
  }

  dat->labels = mmap(0, dat->images, PROT_READ, MAP_PRIVATE, label_file, 0) + sizeof(int32_t) * 2;

  if (dat->labels == MAP_FAILED) {
    fprintf(stderr, "%s:%d: Map failed (%d): %s\n", __FILE__, __LINE__, errno, strerror(errno));
    return FAIL;
  }

  return SUCCESS;
}
void matrix_zero(struct matrix mat) {
  memset(mat.e, 0, mat.x * mat.y * sizeof(float));
}
void matmul(struct matrix a, struct matrix b, struct matrix result) {
  assert(a.y == b.x);
  assert(result.x == a.x);
  assert(result.y == b.y);
  float alpha = 1.0;
  float beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, result.x, result.y,
    a.y, alpha, a.e, a.y, b.e, b.y, beta, result.e, result.y);
}
// a * T(b)
void matmulT(struct matrix a, struct matrix b, struct matrix result) {
  assert(a.y == b.y);
  assert(result.x == a.x);
  assert(result.y == b.x);
  float alpha = 1.0;
  float beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, result.x, result.y, a.y,
    alpha, a.e, a.y, b.e, b.y, beta, result.e, result.y);
}
// T(a) * b
void Tmatmul(struct matrix a, struct matrix b, struct matrix result) {
  assert(a.x == b.x);
  assert(result.x == a.y);
  assert(result.y == b.y);
  float alpha = 1.0;
  float beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, result.x, result.y, a.x,
    alpha, a.e, a.y, b.e, b.y, beta, result.e, result.y);
}
// gaussian distribution with a standard deviation of sigma and an average of mu
// generated using mox-muller transform
void randomize(float *data, size_t count, float sigma) {
  static const float two_pi = 2.0 * 3.14159265358979323846;
  static const float mu = 0.0;

  for (size_t i = 0; i < count; i += 2) {
    float u1, u2;

    do {
      u1 = (float) rand() * (1.0f / RAND_MAX);
      u2 = (float) rand() * (1.0f / RAND_MAX);
    } while (u1 <= FLT_MIN);

    data[i] = sqrtf(-2.0 * logf(u1)) * cosf(two_pi * u2) * sigma + mu;

    if (i + 1 < count) {
      data[i + 1] = sqrtf(-2.0 * logf(u1)) * sinf(two_pi * u2) * sigma + mu;
    }
  }
}
// fisher-yates shuffle
void shuffle(int *data, const size_t count) {
  for (int i = count - 1; i > 0; --i) {
    int j = 0;

    do { // clip the top to make uniform random
      j = rand();
    } while (j > RAND_MAX - (RAND_MAX % i));

    j = j % i;
    // swap
    data[i] ^= data[j];
    data[j] ^= data[i];
    data[i] ^= data[j];
  }
}
float gaussian_activate(float zval, float bias) {
  float sum_term = zval + bias;
  return exp(-1 * sum_term * sum_term);
}
float gaussian_prime(float zval, float activation) {
  return -2 * zval * activation;
}
float tanh_activate(float zval, float bias) {
  float sum_term = zval + bias;
  return tanh(sum_term);
}
float relu_activate(float zval, float bias) {
  float sum_term = zval + bias;
  if(sum_term < 0) {
    return 0;
  } else {
    return sum_term;
  }
}


matrix feedforward(struct neural_layer layer[], const int neural_layers) {
  // propigate the layers
  for (int j = 1; j < neural_layers; ++j) {
    matrix_zero(layer[j].zvals);
    matrix_zero(layer[j].activations);
    matmul(layer[j - 1].activations, layer[j].weights, layer[j].zvals);

    for (int i = 0; i < layer[j].activations.x; ++i) {
      for (int k = 0; k < layer[j].activations.y; ++k) {
        layer[j].activations.e[i * layer[j].activations.y + k] =
          layer[j].activate(layer[j].zvals.e[i * layer[j].zvals.y + k], layer[j].biases.e[k]);
      }
    }
  }

  const int last = neural_layers - 1;
  return layer[last].activations;
}
#endif
