module attributes {torch.debug_module_name = "SimpleConvNet"} {
  func.func @forward(%arg0: tensor<1x1x10x10xf32>) -> tensor<1x16x8x8xf32> {
    %0 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16xf32>}> : () -> tensor<16xf32>
    %1 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16x3x3x1xf32>}> : () -> tensor<16x3x3x1xf32>
    %3 = tosa.reshape %arg0 {new_shape = array<i64: 1, 10, 10, 1>} : (tensor<1x1x10x10xf32>) -> tensor<1x10x10x1xf32>
    %4 = tosa.conv2d %3, %2, %0 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x10x10x1xf32>, tensor<16x3x3x1xf32>, tensor<16xf32>) -> tensor<1x8x8x16xf32>
    %5 = tosa.transpose %4, %1 : (tensor<1x8x8x16xf32>, tensor<4xi32>) -> tensor<1x16x8x8xf32>
    %6 = tosa.clamp %5 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    return %6 : tensor<1x16x8x8xf32>
  }
}