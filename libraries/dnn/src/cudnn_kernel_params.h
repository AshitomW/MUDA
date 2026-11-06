/**
 * @file cudnn_kernel_params.h
 * @brief Parameter structs shared between host C++ and cudnn_kernels.metal.
 */

#pragma once

struct CudnnConvParams {
  int n;
  int c_in;
  int h_in;
  int w_in;
  int x_ns;
  int x_cs;
  int x_hs;
  int x_ws;
  int c_out;
  int h_out;
  int w_out;
  int groups;
  int y_ns;
  int y_cs;
  int y_hs;
  int y_ws;
  int kh;
  int kw;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dil_h;
  int dil_w;
  int w_format;
  int conv_mode;
  float alpha;
  float beta;
};

struct CudnnPoolParams {
  int n;
  int c;
  int h_in;
  int w_in;
  int x_ns;
  int x_cs;
  int x_hs;
  int x_ws;
  int h_out;
  int w_out;
  int y_ns;
  int y_cs;
  int y_hs;
  int y_ws;
  int win_h;
  int win_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int mode;
  float alpha;
  float beta;
  int pad_unused;
};

struct CudnnTensorOpParams {
  int n;
  int c;
  int h;
  int w;
  int x_ns;
  int x_cs;
  int x_hs;
  int x_ws;
  int y_ns;
  int y_cs;
  int y_hs;
  int y_ws;
  int a_n;
  int a_c;
  int a_h;
  int a_w;
  int mode;
  float alpha;
  float beta;
  float coef;
};

