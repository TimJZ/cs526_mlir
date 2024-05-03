 module attributes {torch.debug_module_name = "SimpleConvNet"} {
  func.func @forward(%arg0: tensor<1x1x10x10xf32>) -> tensor<1x10xf32> {
    %0 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<10x256xf32>}> : () -> tensor<10x256xf32>
    %1 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16xf32>}> : () -> tensor<16xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5 = "tosa.const"() <{value = dense<[[0.00256735831, -0.0139928237, 0.0551752597, 0.0247342139, -4.302510e-02, -0.0624865591, 0.014556706, -0.0336430222, -0.0300798491, -2.83122063E-6]]> : tensor<1x10xf32>}> : () -> tensor<1x10xf32>
    %6 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16x3x3x1xf32>}> : () -> tensor<16x3x3x1xf32>
    %7 = tosa.reshape %arg0 {new_shape = array<i64: 1, 10, 10, 1>} : (tensor<1x1x10x10xf32>) -> tensor<1x10x10x1xf32>
    %8 = tosa.conv2d %7, %6, %1 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x10x10x1xf32>, tensor<16x3x3x1xf32>, tensor<16xf32>) -> tensor<1x8x8x16xf32>
    %9 = tosa.transpose %8, %3 : (tensor<1x8x8x16xf32>, tensor<4xi32>) -> tensor<1x16x8x8xf32>
    %10 = tosa.clamp %9 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    %11 = tosa.transpose %10, %2 : (tensor<1x16x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x16xf32>
    %12 = tosa.max_pool2d %11 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x8x8x16xf32>) -> tensor<1x4x4x16xf32>
    %13 = tosa.transpose %12, %3 : (tensor<1x4x4x16xf32>, tensor<4xi32>) -> tensor<1x16x4x4xf32>
    %14 = tosa.transpose %0, %4 : (tensor<10x256xf32>, tensor<2xi32>) -> tensor<256x10xf32>
    %15 = tosa.reshape %13 {new_shape = array<i64: 1, 1, 256>} : (tensor<1x16x4x4xf32>) -> tensor<1x1x256xf32>
    %16 = tosa.reshape %14 {new_shape = array<i64: 1, 256, 10>} : (tensor<256x10xf32>) -> tensor<1x256x10xf32>
    %17 = tosa.matmul %15, %16 : (tensor<1x1x256xf32>, tensor<1x256x10xf32>) -> tensor<1x1x10xf32>
    %18 = tosa.reshape %17 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %19 = tosa.add %18, %5 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %19 : tensor<1x10xf32>
  }
}