// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

// Asking the dimension of a 0-D shape doesn't make sense.
func.func @dim_0_ranked(%arg : tensor<f32>, %arg1 : index) {
  tensor.dim %arg, %arg1 : tensor<f32> // expected-error {{'tensor.dim' op operand #0 must be non-0-ranked or unranked tensor, but got 'tensor<f32>'}}
  return
}

// -----

func.func @tensor.cast_mismatching_constants(%arg0: tensor<1xf32>) {
  // expected-error@+1 {{operand type 'tensor<1xf32>' and result type 'tensor<2xf32>' are cast incompatible}}
  %0 = tensor.cast %arg0 : tensor<1xf32> to tensor<2xf32>
  return
}

// -----

func.func @concat_empty() {
  // expected-error@+1 {{requires at least one input}}
  %0 = tensor.concat dim(0) : () -> tensor<1x2x3xf32>
  return
}

// -----

func.func @concat_rank_mismatch(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
  // expected-error@+1 {{rank of concatenated inputs must match result rank}}
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return
}

// -----

func.func @concat_dim_out_of_range(%arg0: tensor<3xf32>) {
  // expected-error@+1 {{concatenation dim must be less than the tensor rank}}
  %0 = tensor.concat dim(1) %arg0 : (tensor<3xf32>) -> tensor<3xf32>
  return
}

// -----

func.func @concat_element_type_mismatch(%arg0: tensor<3xf32>, %arg1: tensor<3xi32>) {
  // expected-error@+1 {{inputs and result element type must match}}
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<3xf32>, tensor<3xi32>) -> tensor<3xf32>
  return
}

// -----

func.func @concat_incompatible_input_types(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) {
  // expected-error@+1 {{static concatenation size mismatch along non-concatenated dimension 1}}
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<7x5xf32>
  return
}

// -----

func.func @concat_static_shape_mismatch(%arg0: tensor<3xf32>) {
  // expected-error@+1 {{result type 'tensor<7xf32>'does not match inferred shape 'tensor<6xf32>' static sizes}}
  %0 = tensor.concat dim(0) %arg0, %arg0 : (tensor<3xf32>, tensor<3xf32>) -> tensor<7xf32>
  return
}

// -----

func.func @extract_too_many_indices(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = tensor.extract %arg0[] : tensor<?xf32>
  return
}

// -----

func.func @insert_too_many_indices(%arg0: f32, %arg1: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices}}
  %0 = tensor.insert %arg0 into %arg1[] : tensor<?xf32>
  return
}

// -----

func.func @tensor.from_elements_wrong_result_type() {
  // expected-error@+2 {{'tensor.from_elements' invalid kind of type specified}}
  %c0 = arith.constant 0 : i32
  %0 = tensor.from_elements %c0 : tensor<*xi32>
  return
}

// -----

func.func @tensor.from_elements_wrong_elements_count() {
  // expected-error@+2 {{1 operands present, but expected 2}}
  %c0 = arith.constant 0 : index
  %0 = tensor.from_elements %c0 : tensor<2xindex>
  return
}

// -----

func.func @tensor.generate(%m : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have as many index operands as dynamic extents in the result type}}
  %tnsr = tensor.generate %m {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func.func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have one body argument per input dimension}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func.func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{all body arguments must be index}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : i64):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func.func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+4 {{'func.return' op expects parent op 'func.func'}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      func.return %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func.func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{body must be terminated with a `yield` operation of the tensor element type}}
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8 : i32
      tensor.yield %elem : i32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func.func @tensor.reshape_element_type_mismatch(
       %buf: tensor<*xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{element types of source and destination tensor types should be the same}}
  tensor.reshape %buf(%shape) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xi32>
}

// -----

func.func @tensor.reshape_dst_ranked_shape_unranked(
       %buf: tensor<*xf32>, %shape: tensor<?xi32>) {
  // expected-error @+1 {{cannot use shape operand with dynamic length to reshape to statically-ranked tensor type}}
  tensor.reshape %buf(%shape) : (tensor<*xf32>, tensor<?xi32>) -> tensor<?xf32>
}

// -----

func.func @tensor.reshape_dst_shape_rank_mismatch(
       %buf: tensor<*xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{length of shape operand differs from the result's tensor rank}}
  tensor.reshape %buf(%shape)
    : (tensor<*xf32>, tensor<1xi32>) -> tensor<?x?xf32>
}

// -----

func.func @tensor.reshape_num_elements_mismatch(
       %buf: tensor<1xf32>, %shape: tensor<1xi32>) {
  // expected-error @+1 {{source and destination tensor should have the same number of elements}}
  tensor.reshape %buf(%shape)
    : (tensor<1xf32>, tensor<1xi32>) -> tensor<10xf32>
}

// -----

func.func @extract_slice_wrong_result_rank(%t: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected rank to be smaller or equal to the other rank.}}
  %0 = tensor.extract_slice %t[0][4][1] : tensor<?xf32> to tensor<?x?xf32>

  return
}

// -----

func.func @extract_slice_wrong_result_rank(%t: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected element type to be 'f32'}}
  %0 = tensor.extract_slice %t[0][4][1] : tensor<?xf32> to tensor<4xi8>

  return
}

// -----

func.func @extract_slice_wrong_static_type(%t: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<?x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.extract_slice %t[0, 0, 0][%idx, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  return
}

// -----

func.func @extract_slice_wrong_dynamic_type(%t: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<?x4x4xf32>

  return
}

// -----

func.func @insert_slice_wrong_result_rank(%t1: tensor<?xf32>, %t2: tensor<?x?xf32>, %idx : index) {
  // expected-error @+1 {{expected rank to be smaller or equal to the other rank.}}
  %0 = tensor.insert_slice %t2 into %t1[0][4][1] : tensor<?x?xf32> into tensor<?xf32>

  return
}

// -----

func.func @insert_slice_wrong_result_rank(%t1: tensor<4xi8>, %t2: tensor<?xf32>, %idx : index) {
  // expected-error @+1 {{expected element type to be 'f32'}}
  %0 = tensor.insert_slice %t1 into %t2[0][4][1] : tensor<4xi8> into tensor<?xf32>

  return
}

// -----

func.func @insert_slice_wrong_static_type(%t1: tensor<4x4x4xf32>, %t2: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<?x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.insert_slice %t1 into %t2[0, 0, 0][%idx, 4, 4][1, 1, 1]
    : tensor<4x4x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func.func @insert_slice_wrong_dynamic_type(%t1: tensor<?x4x4xf32>, %t2: tensor<8x16x4xf32>, %idx : index) {
  // expected-error @+1 {{expected type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (size mismatch)}}
  %0 = tensor.insert_slice %t1 into %t2[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<?x4x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func.func @illegal_expanding_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x4x?xf32> {
  // expected-error @+1 {{invalid to have a single dimension (2) expanded into multiple dynamic dims (2,4)}}
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<?x?x?xf32> into tensor<?x?x?x4x?xf32>
  return %0 : tensor<?x?x?x4x?xf32>
}

// -----


func.func @illegal_expanding_reshape_static_tensor
    (%arg0: tensor<2x3x20xf32>) -> tensor<2x3x2x4x5xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x20xf32> into tensor<2x3x2x4x5xf32>
  return %0 : tensor<2x3x2x4x5xf32>
}

// -----

func.func @illegal_collapsing_reshape_static_tensor
    (%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x3x20xf32> {
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x2x4x5xf32> into tensor<2x3x20xf32>
  return %0 : tensor<2x3x20xf32>
}

// -----

func.func @illegal_expanding_reshape_mixed_tensor(%arg0 : tensor<?x?xf32>)
    -> tensor<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func.func @illegal_expanding_reshape_mixed_tensor_2(%arg0 : tensor<?x?xf32>)
    -> tensor<?x4x5xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func.func @illegal_collapsing_reshape_mixed_tensor(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_collapsing_reshape_mixed_tensor_2(%arg0 : tensor<?x4x5xf32>)
    -> tensor<?x?xf32> {
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @rank(%0: f32) {
  // expected-error@+1 {{'tensor.rank' op operand #0 must be tensor of any type values}}
  "tensor.rank"(%0): (f32)->index
  return
}

// -----

func.func @illegal_num_offsets(%arg0 : tensor<?x?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected 3 offset values}}
  %0 = tensor.extract_slice %arg0[0, 0] [%arg1, %arg2] [1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  return
}

// -----

func.func @illegal_num_offsets(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : index, %arg3 : index) {
  // expected-error@+1 {{expected 3 offset values}}
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return
}

// -----


func.func @pad_result_type(%arg0: tensor<?x2x3x4xi32>, %arg1: index, %arg2: i32) -> tensor<?x?x?x8xf32> {
  // expected-error @+1 {{specified type 'tensor<?x?x?x8xf32>' does not match the inferred type 'tensor<?x?x?x9xi32>}}
  %0 = tensor.pad %arg0 low[1, %arg1, 2, 2] high[1, 2, %arg1, 3] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %arg2 : i32
  } : tensor<?x2x3x4xi32> to tensor<?x?x?x8xf32>
  return %0 : tensor<?x?x?x8xf32>
}

// -----

func.func @pad_number_of_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{expected the block to have 2 arguments}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func.func @pad_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{op expected block argument 1 to be an index}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: i32, %arg3: i32):
    tensor.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func.func @pad_yield_type(%arg0: tensor<?x4xi32>, %arg1: i8) -> tensor<?x9xi32> {
  // expected-error @+1 {{op expected yield type to match shape element type}}
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %arg1 : i8
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func.func @invalid_splat(%v : f32) {
  // expected-error@+1 {{invalid kind of type specified}}
  tensor.splat %v : memref<8xf32>
  return
}

// -----

func.func @invalid_splat(%v : vector<8xf32>) {
  // expected-error@+1 {{must be integer/index/float type}}
  %w = tensor.splat %v : tensor<8xvector<8xf32>>
  return
}

// -----

func.func @invalid_splat(%v: f32, %m: index) {
  // expected-error@+1 {{incorrect number of dynamic sizes, has 1, expected 2}}
  %w = tensor.splat %v[%m] : tensor<?x8x?xf32>
  return
}

// -----

func.func @gather_empty_dims(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{gather_dims must be non-empty}}
  %out = tensor.gather %source[%indices] gather_dims([]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2xf32>
  return
}

// -----

func.func @gather_coordinate_rank_overflow(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{gather_dims overflow source rank}}
  %out = tensor.gather %source[%indices] gather_dims([0, 1, 2, 3]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2xf32>
  return
}

// -----

func.func @gather_coordinate_negative(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{gather_dims value must be non-negative}}
  %out = tensor.gather %source[%indices] gather_dims([-1]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @gather_coordinate_overflow(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{gather_dims value must be smaller than source rank}}
  %out = tensor.gather %source[%indices] gather_dims([42]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @gather_coordinate_overflow(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{gather_dims values must be strictly increasing}}
  %out = tensor.gather %source[%indices] gather_dims([1, 0]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @gather_wrong_result_type(
    %source : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{result type mismatch: expected 'tensor<1x2x1x5x1xf32>' or its rank-reduced variant 'tensor<1x2x5xf32>' (got: 'tensor<1x2x1xf32>')}}
  %out = tensor.gather %source[%indices] gather_dims([0, 2]):
    (tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1xf32>
  return
}

// -----

func.func @scatter_empty_dims(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{scatter_dims must be non-empty}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2xf32>
  return
}

// -----

func.func @scatter_coordinate_rank_overflow(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{scatter_dims overflow dest rank}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([0, 1, 2, 3]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2xf32>
  return
}

// -----

func.func @scatter_coordinate_negative(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{scatter_dims value must be non-negative}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([-1]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @scatter_coordinate_overflow(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{scatter_dims value must be smaller than dest rank}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([42]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @scatter_coordinate_overflow(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{scatter_dims values must be strictly increasing}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([1, 0]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32>
  return
}

// -----

func.func @scatter_missing_unique(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{requires 'unique' attribute to be set}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([0, 2]):
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1xf32>
  return
}

// -----

func.func @scatter_wrong_result_type(
    %source : tensor<f32>,
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x2x3xindex>) {
  // expected-error@+1 {{source type mismatch: expected 'tensor<1x2x1x5x1xf32>' or its rank-reduced variant 'tensor<1x2x5xf32>' (got: 'tensor<f32>')}}
  %out = tensor.scatter %source into %dest[%indices] scatter_dims([0, 2]) unique:
    (tensor<f32>, tensor<4x5x6xf32>, tensor<1x2x3xindex>) -> tensor<1x2x1xf32>
  return
}

// -----

func.func @empty_wrong_number_of_operands(%sz : index) {
  // expected-error@+1 {{incorrect number of dynamic sizes, has 1, expected 2}}
  %out = tensor.empty(%sz) : tensor<2x?x?x5xf32>
  return
}

// -----

func.func @pack_invalid_no_padding_no_full_tiles(%input: tensor<256x128xf32>, %output: tensor<8x8x16x33xf32>) -> tensor<8x8x16x33xf32> {
  // expected-error@+1 {{invalid tile factor or output size provided. Only full tiles are supported when padding_value is not set}}
  %0 = tensor.pack %input inner_dims_pos = [1, 0] inner_tiles = [16, 33] into %output : tensor<256x128xf32>  -> tensor<8x8x16x33xf32>
  return %0 : tensor<8x8x16x33xf32>
}

// -----

func.func @pack_invalid_no_padding_no_full_tiles_dyn_tiles(%input: tensor<256x128xf32>, %output: tensor<10x8x?x?xf32>, %tile_size_0: index, %tile_size_1: index) -> tensor<10x8x?x?xf32> {
  // expected-error@+1 {{invalid tile factor or output size provided. Only full tiles are supported when padding_value is not set}}
  %0 = tensor.pack %input inner_dims_pos = [1, 0] inner_tiles = [%tile_size_0, %tile_size_1] into %output : tensor<256x128xf32>  -> tensor<10x8x?x?xf32>
  return %0 : tensor<10x8x?x?xf32>
} 

// -----

func.func @pack_invalid_no_padding_no_full_tiles_dyn_tiles_outperm(%input: tensor<256x128xf32>, %output: tensor<8x10x?x?xf32>, %tile_size_0: index, %tile_size_1: index) -> tensor<8x10x?x?xf32> {
  // expected-error@+1 {{invalid tile factor or output size provided. Only full tiles are supported when padding_value is not set}}
  %0 = tensor.pack %input outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [%tile_size_0, %tile_size_1] into %output : tensor<256x128xf32>  -> tensor<8x10x?x?xf32>
  return %0 : tensor<8x10x?x?xf32>
} 

// -----

func.func @pad_and_pack_invalid_type(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: i32) -> tensor<2x8x8x2xf32> {
  // expected-error@+1 {{expected padding_value has 'f32' but got: 'i32'}}
  %0 = tensor.pack %input padding_value(%pad: i32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

// -----

func.func @pack_invalid_inner_dims_pos_vector(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid inner_dims_pos vector}}
  %0 = tensor.pack %input inner_dims_pos = [2, 0] inner_tiles = [2, 2] into %output : tensor<256x128xf32> -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @pack_invalid_duplicate_element_in_inner_dims(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid inner_dims_pos vector}}
  %0 = tensor.pack %input inner_dims_pos = [1, 1] inner_tiles = [2, 2] into %output : tensor<256x128xf32> -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @pack_invalid_duplicate_element_in_outer_perm(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid outer_dims_perm vector}}
  %0 = tensor.pack %input outer_dims_perm = [1, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %output : tensor<256x128xf32> -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @unpack_invalid_out_of_bound_outer_perm(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid outer_dims_perm vector}}
  %0 = tensor.unpack %output outer_dims_perm = [2, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %input : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// -----

func.func @pack_invalid_outer_dims_perm(%source: tensor<128x256xf32>, %dest: tensor<16x4x32x16xf32>) -> tensor<16x4x32x16xf32> {
  // expected-error@+1 {{outer_dims_perm must be a permutation or empty}}
  %0 = tensor.pack %source outer_dims_perm = [0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<16x4x32x16xf32>
  return %0 : tensor<16x4x32x16xf32>
}

// -----

func.func @unpack_invalid_outer_dims_perm(%source: tensor<128x256xf32>, %dest: tensor<16x4x32x16xf32>) -> tensor<128x256xf32> {
  // expected-error@+1 {{outer_dims_perm must be a permutation or empty}}
  %0 = tensor.unpack %dest outer_dims_perm = [1] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %source : tensor<16x4x32x16xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{the shape of output is not large enough to hold the packed data. Expected at least 'tensor<8x8x16x32xf32>', got 'tensor<8x8x32x16xf32>'}}
  %0 = tensor.pack %input inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %output : tensor<256x128xf32> -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @unpack_invalid(%output: tensor<256x128xf32>, %input: tensor<8x8x32x16xf32>) -> tensor<256x128xf32> {
  // expected-error@+1 {{the shape of output is not large enough to hold the packed data. Expected at least 'tensor<8x32x4x32xf32>', got 'tensor<8x8x32x16xf32>'}}
  %0 = tensor.unpack %input inner_dims_pos = [1, 0] inner_tiles = [4, 32] into %output : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid zero tile factor}}
  %0 = tensor.pack %input inner_dims_pos = [1, 0] inner_tiles = [0, 2] into %output : tensor<256x128xf32> -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----
func.func @pack_mismatch_inner_tile_size_and_output_shape(
  %input : tensor<?x?xf32>, %output : tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32> {
  // expected-error@+1 {{mismatch in inner tile sizes specified and shaped of tiled dimension in the packed type}}
  %0 = tensor.pack %input inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %output : tensor<?x?xf32> -> tensor<?x?x8x8xf32>
  return %0 : tensor<?x?x8x8xf32>
}

// -----

func.func @unpack_mismatch_inner_tile_size_and_output_shape(
  %input : tensor<?x?x8x8xf32>, %output : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch in inner tile sizes specified and shaped of tiled dimension in the packed type}}
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %output : tensor<?x?x8x8xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
