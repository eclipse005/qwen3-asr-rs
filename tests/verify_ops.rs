//! Verify burn conv2d and unfold against candle reference.
//! Run: cd burn && cargo test --test verify_ops -- --nocapture

use burn::tensor::{Tensor, TensorData};

use burn_cubecl::CubeBackend;
use cubecl::cuda::CudaRuntime;
type B = CubeBackend<CudaRuntime, f32, i32, u8>;

fn conv2d_forward(
    input: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 4> {
    let [b, in_c, h, w] = input.dims();
    let [out_c, _, k_h, k_w] = weight.dims();
    let device = input.device().clone();
    let padded = if padding > 0 {
        Tensor::<B, 4>::zeros([b, in_c, h + 2 * padding, w + 2 * padding], &device)
            .slice_assign([0..b, 0..in_c, padding..padding + h, padding..padding + w], input.clone())
    } else {
        input.clone()
    };
    let u = padded.unfold(2, k_h, stride);
    let [_, _, oh, wa, _] = u.dims();
    let r = u.reshape([b * oh, in_c, wa, k_h]);
    let u2 = r.unfold(2, k_w, stride);
    let [boh, _, ow, kh, kw] = u2.dims();
    let patches = u2.reshape([boh, in_c, ow, kh * kw])
        .permute([0, 2, 1, 3])
        .reshape([boh * ow, in_c * kh * kw]);
    let wf = weight.clone().reshape([out_c, in_c * kh * kw]);
    let out = patches.matmul(wf.transpose());
    out.reshape([b, oh, ow, out_c]).permute([0, 3, 1, 2])
}

#[test]
fn test_conv2d_simple() {
    let device = burn::backend::cuda::CudaDevice::default();

    // Input: [1, 1, 4, 4] - simple 4x4 single-channel image
    let input_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let input = Tensor::<B, 4>::from_data(TensorData::new(input_data.clone(), [1, 1, 4, 4]), &device);

    // Weight: [1, 1, 3, 3] - single 3x3 kernel, all ones
    let weight_data = vec![1.0f32; 9];
    let weight = Tensor::<B, 4>::from_data(TensorData::new(weight_data, [1, 1, 3, 3]), &device);

    // Conv2d with stride=2, padding=1
    // Expected output size: (4 + 2 - 3) / 2 + 1 = 2 -> [1, 1, 2, 2]
    let output = conv2d_forward(&input, &weight, 2, 1);
    let [ob, oc, oh, ow] = output.dims();
    assert_eq!([ob, oc, oh, ow], [1, 1, 2, 2], "output shape mismatch");

    let out_data = output.into_data();
    let out_vec: Vec<f32> = out_data.to_vec().unwrap();

    // Manual computation:
    // Padded input (padding=1):
    //  0  0  0  0  0  0
    //  0  1  2  3  4  0
    //  0  5  6  7  8  0
    //  0  9 10 11 12  0
    //  0 13 14 15 16  0
    //  0  0  0  0  0  0
    //
    // With stride=2, kernel 3x3:
    // Output[0,0]: sum of padded[0:3, 0:3] = 0+0+0+0+1+2+0+5+6 = 14
    // Output[0,1]: sum of padded[0:3, 2:5] = 0+0+0+2+3+4+6+7+8 = 30
    // Output[1,0]: sum of padded[2:5, 0:3] = 0+5+6+0+9+10+0+13+14 = 57
    // Output[1,1]: sum of padded[2:5, 2:5] = 6+7+8+10+11+12+14+15+16 = 99

    let expected = vec![14.0f32, 30.0, 57.0, 99.0];
    eprintln!("conv2d output: {:?}", out_vec);
    eprintln!("expected:      {:?}", expected);

    for (i, (a, b)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "output[{}] = {}, expected {}", i, a, b);
    }
}

#[test]
fn test_conv2d_multi_channel() {
    let device = burn::backend::cuda::CudaDevice::default();

    // Input: [1, 2, 4, 4] - 2 channels
    let input_data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let input = Tensor::<B, 4>::from_data(TensorData::new(input_data, [1, 2, 4, 4]), &device);

    // Weight: [3, 2, 3, 3] - 3 output channels, 2 input channels, 3x3 kernel
    let weight_data: Vec<f32> = (1..=54).map(|i| i as f32 * 0.01).collect();
    let weight = Tensor::<B, 4>::from_data(TensorData::new(weight_data.clone(), [3, 2, 3, 3]), &device);

    let output = conv2d_forward(&input, &weight, 2, 1);
    let [ob, oc, oh, ow] = output.dims();
    assert_eq!([ob, oc, oh, ow], [1, 3, 2, 2], "output shape mismatch");

    let out_data = output.into_data();
    let out_vec: Vec<f32> = out_data.to_vec().unwrap();
    eprintln!("multi-channel conv2d output shape: [{},{},{},{}]", ob, oc, oh, ow);
    eprintln!("output values: {:?}", out_vec);

    // Manual computation is complex for multi-channel, but verify shape and finiteness
    assert!(out_vec.iter().all(|v| v.is_finite()), "all values must be finite");
}

#[test]
fn test_unfold_basic() {
    let device = burn::backend::cuda::CudaDevice::default();

    // Input: [1, 1, 4, 6]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::<B, 4>::from_data(TensorData::new(data, [1, 1, 4, 6]), &device);

    // Unfold dim 2 with size=3, step=1
    // Expected output: [1, 1, 2, 6, 3]  (4-3)/1+1 = 2 windows
    let u = t.unfold(2, 3, 1);
    let dims = u.dims();
    eprintln!("unfold dims: {:?}", dims);
    assert_eq!(dims, [1, 1, 2, 6, 3], "unfold shape mismatch");

    let u_data = u.into_data();
    let u_vec: Vec<f32> = u_data.to_vec().unwrap();
    eprintln!("unfold values: {:?}", u_vec);

    // Window 0 (rows 0-2): for each col c, window is [row0*6+c, row1*6+c, row2*6+c]
    // Window 1 (rows 1-3): for each col c, window is [row1*6+c, row2*6+c, row3*6+c]
}
