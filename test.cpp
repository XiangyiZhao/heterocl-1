#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(ap_uint<24> RGB[412][280], float Gx[3][3], float Gy[3][3], float Fimg[410][278]) {
  #pragma HLS array_partition variable=Gy complete dim=0
  #pragma HLS array_partition variable=Gx complete dim=0
  float _top;
  float B[412][280];
  B_x: for (ap_int<32> x = 0; x < 412; ++x) {
    B_y: for (ap_int<32> y = 0; y < 280; ++y) {
    #pragma HLS pipeline
      B[x][y] = ((float)(((ap_uint<10>)(((ap_uint<9>)((ap_uint<8>)RGB[x][y](7, 0))) + ((ap_uint<9>)((ap_uint<8>)RGB[x][y](15, 8))))) + ((ap_uint<10>)((ap_uint<8>)RGB[x][y](23, 16)))));
    }
  }
  float xx[410][278];
  float LBX[3][280];
  #pragma HLS array_partition variable=LBX complete dim=1
  float WBX[3][3];
  #pragma HLS array_partition variable=WBX complete dim=0
  xx_x_reuse: for (ap_int<32> x_reuse = 0; x_reuse < 412; ++x_reuse) {
    xx_y_reuse: for (ap_int<32> y_reuse = 0; y_reuse < 280; ++y_reuse) {
    #pragma HLS pipeline
      B_1: for (ap_int<32> B_1 = 0; B_1 < 2; ++B_1) {
        LBX[B_1][y_reuse] = LBX[(B_1 + 1)][y_reuse];
      }
      LBX[2][y_reuse] = B[x_reuse][y_reuse];
      if (2 <= x_reuse) {
        LBX_1: for (ap_int<32> LBX_1 = 0; LBX_1 < 3; ++LBX_1) {
          LBX_0: for (ap_int<32> LBX_0 = 0; LBX_0 < 2; ++LBX_0) {
            WBX[LBX_1][LBX_0] = WBX[LBX_1][(LBX_0 + 1)];
          }
          WBX[LBX_1][2] = LBX[LBX_1][y_reuse];
        }
        if (2 <= y_reuse) {
          ap_int<32> sum1;
          sum1_ra0: for (ap_int<32> ra0 = 0; ra0 < 3; ++ra0) {
            sum1_ra1: for (ap_int<32> ra1 = 0; ra1 < 3; ++ra1) {
              sum1 = ((ap_int<32>)((WBX[ra0][ra1] * Gx[ra0][ra1]) + ((float)sum1)));
            }
          }
          xx[(x_reuse + -2)][(y_reuse + -2)] = ((float)sum1);
        }
      }
    }
  }
  float yy[410][278];
  float LBY[3][280];
  #pragma HLS array_partition variable=LBY complete dim=1
  float WBY[3][3];
  #pragma HLS array_partition variable=WBY complete dim=0
  yy_x_reuse1: for (ap_int<32> x_reuse1 = 0; x_reuse1 < 412; ++x_reuse1) {
    yy_y_reuse1: for (ap_int<32> y_reuse1 = 0; y_reuse1 < 280; ++y_reuse1) {
    #pragma HLS pipeline
      B_11: for (ap_int<32> B_11 = 0; B_11 < 2; ++B_11) {
        LBY[B_11][y_reuse1] = LBY[(B_11 + 1)][y_reuse1];
      }
      LBY[2][y_reuse1] = B[x_reuse1][y_reuse1];
      if (2 <= x_reuse1) {
        LBY_1: for (ap_int<32> LBY_1 = 0; LBY_1 < 3; ++LBY_1) {
          LBY_0: for (ap_int<32> LBY_0 = 0; LBY_0 < 2; ++LBY_0) {
            WBY[LBY_1][LBY_0] = WBY[LBY_1][(LBY_0 + 1)];
          }
          WBY[LBY_1][2] = LBY[LBY_1][y_reuse1];
        }
        if (2 <= y_reuse1) {
          ap_int<32> sum2;
          sum2_ra2: for (ap_int<32> ra2 = 0; ra2 < 3; ++ra2) {
            sum2_ra3: for (ap_int<32> ra3 = 0; ra3 < 3; ++ra3) {
              sum2 = ((ap_int<32>)((WBY[ra2][ra3] * Gy[ra2][ra3]) + ((float)sum2)));
            }
          }
          yy[(x_reuse1 + -2)][(y_reuse1 + -2)] = ((float)sum2);
        }
      }
    }
  }
  Fimg_x1: for (ap_int<32> x1 = 0; x1 < 410; ++x1) {
    Fimg_y1: for (ap_int<32> y1 = 0; y1 < 278; ++y1) {
    #pragma HLS pipeline
      Fimg[x1][y1] = (sqrt(((xx[x1][y1] * xx[x1][y1]) + (yy[x1][y1] * yy[x1][y1]))) * 5.891867e-02f);
    }
  }
}

