module attributes {torch.debug_module_name = "LeNet"} {
  func.func @forward(%arg0: tensor<1x1x32x32xf32>) -> tensor<1x10xf32> {
    %0 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<10x120xf32>}> : () -> tensor<10x120xf32>
    %1 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<120xf32>}> : () -> tensor<120xf32>
    %2 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<120x16x5x5xf32>}> : () -> tensor<120x16x5x5xf32>
    %3 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16xf32>}> : () -> tensor<16xf32>
    %4 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<16x6x5x5xf32>}> : () -> tensor<16x6x5x5xf32>
    %5 = "tosa.const"() <{value = dense<[0.104563646, -0.120800331, 0.141011402, -0.108872436, -0.149258777, -0.167770311]> : tensor<6xf32>}> : () -> tensor<6xf32>
    %6 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %7 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %8 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %9 = "tosa.const"() <{value = dense<[[-0.00235931948, -0.0843035802, 0.0113061666, 0.0434347056, 0.0641338155, -0.0235007945, 0.0579165891, -0.0158394594, 0.0536150895, -0.0531563386]]> : tensor<1x10xf32>}> : () -> tensor<1x10xf32>
    %10 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<6x5x5x1xf32>}> : () -> tensor<6x5x5x1xf32>
    %11 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32, 32, 1>} : (tensor<1x1x32x32xf32>) -> tensor<1x32x32x1xf32>
    %12 = tosa.conv2d %11, %10, %5 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x1xf32>, tensor<6x5x5x1xf32>, tensor<6xf32>) -> tensor<1x28x28x6xf32>
    %13 = tosa.transpose %12, %7 : (tensor<1x28x28x6xf32>, tensor<4xi32>) -> tensor<1x6x28x28xf32>
    %14 = tosa.clamp %13 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x6x28x28xf32>) -> tensor<1x6x28x28xf32>
    %15 = tosa.transpose %14, %6 : (tensor<1x6x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x6xf32>
    %16 = tosa.max_pool2d %15 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x6xf32>) -> tensor<1x14x14x6xf32>
    %17 = tosa.transpose %16, %7 : (tensor<1x14x14x6xf32>, tensor<4xi32>) -> tensor<1x6x14x14xf32>
    %18 = tosa.clamp %17 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %19 = tosa.transpose %18, %6 : (tensor<1x6x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x6xf32>
    %20 = tosa.transpose %4, %6 : (tensor<16x6x5x5xf32>, tensor<4xi32>) -> tensor<16x5x5x6xf32>
    %21 = tosa.conv2d %19, %20, %3 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x6xf32>, tensor<16x5x5x6xf32>, tensor<16xf32>) -> tensor<1x10x10x16xf32>
    %22 = tosa.transpose %21, %7 : (tensor<1x10x10x16xf32>, tensor<4xi32>) -> tensor<1x16x10x10xf32>
    %23 = tosa.clamp %22 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x10x10xf32>) -> tensor<1x16x10x10xf32>
    %24 = tosa.transpose %23, %6 : (tensor<1x16x10x10xf32>, tensor<4xi32>) -> tensor<1x10x10x16xf32>
    %25 = tosa.max_pool2d %24 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x10x10x16xf32>) -> tensor<1x5x5x16xf32>
    %26 = tosa.transpose %25, %7 : (tensor<1x5x5x16xf32>, tensor<4xi32>) -> tensor<1x16x5x5xf32>
    %27 = tosa.clamp %26 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
    %28 = tosa.transpose %27, %6 : (tensor<1x16x5x5xf32>, tensor<4xi32>) -> tensor<1x5x5x16xf32>
    %29 = tosa.transpose %2, %6 : (tensor<120x16x5x5xf32>, tensor<4xi32>) -> tensor<120x5x5x16xf32>
    %30 = tosa.conv2d %28, %29, %1 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x5x5x16xf32>, tensor<120x5x5x16xf32>, tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %31 = tosa.reshape %30 {new_shape = array<i64: 1, 120, 1, 1>} : (tensor<1x1x1x120xf32>) -> tensor<1x120x1x1xf32>
    %32 = tosa.clamp %31 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %33 = tosa.transpose %0, %8 : (tensor<10x120xf32>, tensor<2xi32>) -> tensor<120x10xf32>
    %34 = tosa.reshape %32 {new_shape = array<i64: 1, 1, 120>} : (tensor<1x120x1x1xf32>) -> tensor<1x1x120xf32>
    %35 = tosa.reshape %33 {new_shape = array<i64: 1, 120, 10>} : (tensor<120x10xf32>) -> tensor<1x120x10xf32>
    %36 = tosa.matmul %34, %35 : (tensor<1x1x120xf32>, tensor<1x120x10xf32>) -> tensor<1x1x10xf32>
    %37 = tosa.reshape %36 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %38 = tosa.add %37, %9 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %39 = tosa.clamp %38 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %39 : tensor<1x10xf32>
  }
}
