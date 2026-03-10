import Foundation
import MLX
import MLXNN

// MARK: - Metal Availability Check

/// Check whether the Metal GPU backend is available.
private func _hasMetal() -> Bool {
    // On macOS/iOS with MLX, Metal should always be available.
    // We check by verifying the default device is GPU.
    let device = Device.defaultDevice()
    return device.deviceType == .gpu
}

let hasMetal: Bool = _hasMetal()

/// Threshold: when elements per group exceeds this, use pure MLX ops
/// instead of the single-threadgroup Metal kernel (which underutilizes the GPU).
private let hybridThreshold = 32768

// MARK: - Fused GLU Kernel

/// Metal source for fused GLU: a * sigmoid(b) where [a, b] = split(x, 2, axis)
///
/// After reshaping to (N, 2*C) with target axis last, the memory layout is:
///   row i: [a_0, a_1, ..., a_{C-1}, b_0, b_1, ..., b_{C-1}]
/// So for element (row, col) where col < C:
///   a = x[row * 2C + col]
///   b = x[row * 2C + C + col]
private let gluSource = """
uint gid = thread_position_in_grid.x;
uint total = params[0];     // N * C (total output elements)
uint half_dim = params[1];  // C (half of last dimension)

if (gid >= total) return;

uint row = gid / half_dim;
uint col = gid % half_dim;
uint full_dim = half_dim * 2;

float a = (float)x[row * full_dim + col];
float b = (float)x[row * full_dim + half_dim + col];
float sig_b = 1.0f / (1.0f + metal::exp(-b));
out[gid] = (T)(a * sig_b);
"""

nonisolated(unsafe) private var _gluKernel: MLXFast.MLXFastKernel? = nil

private func getGLUKernel() -> MLXFast.MLXFastKernel {
    if let k = _gluKernel { return k }
    let k = MLXFast.metalKernel(
        name: "fused_glu",
        inputNames: ["x", "params"],
        outputNames: ["out"],
        source: gluSource
    )
    _gluKernel = k
    return k
}

/// Fused GLU: split x in half along axis, compute a * sigmoid(b).
///
/// Equivalent to:
///     let parts = split(x, parts: 2, axis: axis)
///     return parts[0] * sigmoid(parts[1])
func fusedGLU(_ x: MLXArray, axis: Int = 1) -> MLXArray {
    if !hasMetal {
        let parts = split(x, parts: 2, axis: axis)
        return parts[0] * sigmoid(parts[1])
    }

    let ndim = x.ndim
    let normalizedAxis = ((axis % ndim) + ndim) % ndim

    let inShape = x.shape
    guard inShape[normalizedAxis] % 2 == 0 else {
        fatalError("Axis \(normalizedAxis) size must be even, got \(inShape[normalizedAxis])")
    }

    // Move target axis to last for contiguous split
    var xT = x
    if normalizedAxis != ndim - 1 {
        var perm = Array(0..<ndim)
        perm.swapAt(normalizedAxis, ndim - 1)
        xT = xT.transposed(axes: perm)
    }

    let lastDim = xT.dim(-1)
    let half = lastDim / 2
    let x2d = contiguous(xT.reshaped([-1, lastDim]))
    let n = x2d.dim(0)
    let total = n * half

    let params = MLXArray([Int32(total), Int32(half)])

    let resultFlat = getGLUKernel()(
        [x2d.reshaped([-1]), params],
        template: [("T", x.dtype)],
        grid: (total, 1, 1),
        threadGroup: (min(256, total), 1, 1),
        outputShapes: [[total]],
        outputDTypes: [x.dtype]
    )[0]

    var outShape = xT.shape
    outShape[outShape.count - 1] = half
    var result = resultFlat.reshaped(outShape)

    if normalizedAxis != ndim - 1 {
        var perm = Array(0..<ndim)
        perm.swapAt(normalizedAxis, ndim - 1)
        result = result.transposed(axes: perm)
    }

    return result
}

// MARK: - Fused GroupNorm + GELU Kernel

/// Header with erf approximation for GroupNorm+GELU kernel
private let groupNormGELUHeader = """
// Abramowitz & Stegun approximation of erf, max error ~1.5e-7
inline float erf_approx(float x) {
    // erf(-x) = -erf(x)
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float a = metal::abs(x);
    // A&S formula 7.1.26
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float poly = 0.254829592f * t
               - 0.284496736f * t2
               + 1.421413741f * t3
               - 1.453152027f * t4
               + 1.061405429f * t5;
    float result = 1.0f - poly * metal::exp(-a * a);
    return sign * result;
}
"""

/// Metal source for fused GroupNorm + GELU.
/// Each threadgroup handles one (batch, group) pair.
/// Uses simdgroup reductions for mean/variance.
/// GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
private let groupNormGELUSource = """
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Pass 3: Normalize, apply affine, apply erf-based GELU
float rsqrt2 = 0.7071067811865475f;  // 1/sqrt(2)
for (uint i = tid; i < elems_per_group; i += tg_size) {
    uint c_local = i / spatial_size;
    uint c_global = group_idx * channels_per_group + c_local;
    float val = ((float)x[base + i] - mean) * inv_std;
    val = val * (float)weight[c_global] + (float)bias[c_global];
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    val = 0.5f * val * (1.0f + erf_approx(val * rsqrt2));
    out[base + i] = (T)val;
}
"""

nonisolated(unsafe) private var _groupNormGELUKernel: MLXFast.MLXFastKernel? = nil

private func getGroupNormGELUKernel() -> MLXFast.MLXFastKernel {
    if let k = _groupNormGELUKernel { return k }
    let k = MLXFast.metalKernel(
        name: "fused_groupnorm_gelu",
        inputNames: ["x", "weight", "bias", "eps", "params"],
        outputNames: ["out"],
        source: groupNormGELUSource,
        header: groupNormGELUHeader
    )
    _groupNormGELUKernel = k
    return k
}

/// Pure-MLX GroupNorm + GELU fallback (no Metal required).
private func groupNormGELUFallback(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray,
    numGroups: Int, eps: Float = 1e-5
) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)
    let cpg = c / numGroups

    // Reshape to [B, numGroups, cpg, ...spatial...]
    var reshapedDims = [b, numGroups, cpg]
    for i in 2..<x.ndim {
        reshapedDims.append(x.dim(i))
    }
    let xR = x.reshaped(reshapedDims)

    // Axes for mean/var: all axes from index 2 onward
    let axes = Array(2..<xR.ndim)
    let m = mean(xR, axes: axes, keepDims: true)
    let v = variance(xR, axes: axes, keepDims: true)
    let xNorm = (xR - m) * rsqrt(v + MLXArray(eps))
    let xOut = xNorm.reshaped(x.shape)

    // Weight/bias broadcast shape: [1, C, 1, 1, ...]
    var wShape = [Int](repeating: 1, count: x.ndim)
    wShape[1] = c
    let xAffine = xOut * weight.reshaped(wShape) + bias.reshaped(wShape)
    return gelu(xAffine)
}

/// Fused GroupNorm + GELU for NCL or NCHW layout.
///
/// Equivalent to:
///     x = groupNorm(x, weight, bias, numGroups, eps)
///     x = gelu(x)
func fusedGroupNormGELU(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    numGroups: Int,
    eps: Float = 1e-5
) -> MLXArray {
    if !hasMetal {
        return groupNormGELUFallback(x, weight: weight, bias: bias, numGroups: numGroups, eps: eps)
    }

    let origShape = x.shape
    let b = x.dim(0)
    let c = x.dim(1)
    var spatialSize = 1
    for i in 2..<x.ndim {
        spatialSize *= x.dim(i)
    }

    guard c % numGroups == 0 else {
        fatalError("channels \(c) not divisible by numGroups \(numGroups)")
    }
    let channelsPerGroup = c / numGroups
    let elemsPerGroup = channelsPerGroup * spatialSize

    // For large groups, use pure MLX ops which parallelize better
    if elemsPerGroup > hybridThreshold {
        return groupNormGELUFallback(x, weight: weight, bias: bias, numGroups: numGroups, eps: eps)
    }

    let xContig = contiguous(x.reshaped([b, c, spatialSize]))
    let wF32 = contiguous(weight.asType(.float32))
    let bF32 = contiguous(bias.asType(.float32))
    let epsArr = MLXArray([eps])
    let params = MLXArray([Int32(numGroups), Int32(channelsPerGroup), Int32(spatialSize), Int32(c)])

    let totalGroups = b * numGroups
    let tg = min(1024, max(32, ((elemsPerGroup + 31) / 32) * 32))

    let result = getGroupNormGELUKernel()(
        [xContig, wF32, bF32, epsArr, params],
        template: [("T", x.dtype)],
        grid: (totalGroups * tg, 1, 1),
        threadGroup: (tg, 1, 1),
        outputShapes: [[b, c, spatialSize]],
        outputDTypes: [x.dtype]
    )[0]

    return result.reshaped(origShape)
}

// MARK: - Fused GroupNorm + GLU Kernel

/// Metal source for fused GroupNorm + GLU.
/// Each threadgroup handles one (batch, group) pair.
/// GroupNorm normalizes over 2C channels, then GLU splits into a (first C) and
/// b (second C), computing output = a * sigmoid(b). Output has half the channels.
private let groupNormGLUSource = """
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];
uint half_channels = params[4];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Pass 3: Normalize, apply affine, then apply GLU
// Half channels per group on the output side:
uint half_cpg = channels_per_group / 2;
uint out_epg = half_cpg * spatial_size;
uint out_base = batch_idx * half_channels * spatial_size
              + group_idx * half_cpg * spatial_size;

for (uint i = tid; i < out_epg; i += tg_size) {
    uint c_local = i / spatial_size;
    uint s = i % spatial_size;

    // 'a' channel: first half of the group
    uint c_a = c_local;
    uint c_a_global = group_idx * channels_per_group + c_a;
    float val_a = ((float)x[base + c_a * spatial_size + s] - mean) * inv_std;
    val_a = val_a * (float)weight[c_a_global] + (float)bias[c_a_global];

    // 'b' channel: second half of the group
    uint c_b = c_local + half_cpg;
    uint c_b_global = group_idx * channels_per_group + c_b;
    float val_b = ((float)x[base + c_b * spatial_size + s] - mean) * inv_std;
    val_b = val_b * (float)weight[c_b_global] + (float)bias[c_b_global];

    // GLU: a * sigmoid(b)
    float sig_b = 1.0f / (1.0f + metal::exp(-val_b));
    out[out_base + i] = (T)(val_a * sig_b);
}
"""

nonisolated(unsafe) private var _groupNormGLUKernel: MLXFast.MLXFastKernel? = nil

private func getGroupNormGLUKernel() -> MLXFast.MLXFastKernel {
    if let k = _groupNormGLUKernel { return k }
    let k = MLXFast.metalKernel(
        name: "fused_groupnorm_glu",
        inputNames: ["x", "weight", "bias", "eps", "params"],
        outputNames: ["out"],
        source: groupNormGLUSource
    )
    _groupNormGLUKernel = k
    return k
}

/// Pure-MLX GroupNorm + GLU fallback (no Metal required).
private func groupNormGLUFallback(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray,
    numGroups: Int, eps: Float = 1e-5
) -> MLXArray {
    let b = x.dim(0)
    let cFull = x.dim(1)
    let cpg = cFull / numGroups

    // Reshape to [B, numGroups, cpg, ...spatial...]
    var reshapedDims = [b, numGroups, cpg]
    for i in 2..<x.ndim {
        reshapedDims.append(x.dim(i))
    }
    let xR = x.reshaped(reshapedDims)

    let axes = Array(2..<xR.ndim)
    let m = mean(xR, axes: axes, keepDims: true)
    let v = variance(xR, axes: axes, keepDims: true)
    let xNorm = (xR - m) * rsqrt(v + MLXArray(eps))
    let xOut = xNorm.reshaped(x.shape)

    var wShape = [Int](repeating: 1, count: x.ndim)
    wShape[1] = cFull
    let normed = xOut * weight.reshaped(wShape) + bias.reshaped(wShape)

    let parts = split(normed, parts: 2, axis: 1)
    return parts[0] * sigmoid(parts[1])
}

/// Fused GroupNorm + GLU for NCL or NCHW layout.
///
/// GroupNorm normalizes over 2C channels, then GLU splits in half on axis=1.
///
/// Equivalent to:
///     x = groupNorm(x, weight, bias, numGroups, eps)
///     let parts = split(x, parts: 2, axis: 1)
///     return parts[0] * sigmoid(parts[1])
///
/// Input shape: (B, 2C, ...) -> Output shape: (B, C, ...)
///
/// Note: Only supported when numGroups=1 or when channelsPerGroup is even
/// (so the GLU split aligns with group boundaries). When numGroups > 1 and
/// channelsPerGroup is odd, falls back to separate GroupNorm + GLU.
func fusedGroupNormGLU(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    numGroups: Int,
    eps: Float = 1e-5
) -> MLXArray {
    let origShape = x.shape
    let b = x.dim(0)
    let cFull = x.dim(1)  // 2C
    let cHalf = cFull / 2  // C

    var spatialSize = 1
    for i in 2..<x.ndim {
        spatialSize *= x.dim(i)
    }

    guard cFull % numGroups == 0 else {
        fatalError("channels \(cFull) not divisible by numGroups \(numGroups)")
    }
    guard cFull % 2 == 0 else {
        fatalError("channels \(cFull) must be even for GLU")
    }
    let channelsPerGroup = cFull / numGroups
    let elemsPerGroup = channelsPerGroup * spatialSize

    // Pure-MLX fallback for cases where the Metal kernel underperforms:
    // - No Metal available
    // - Large groups (few threadgroups -> GPU underutilization)
    // - numGroups > 1 (GLU split crosses group boundaries)
    if !hasMetal || numGroups > 1 || elemsPerGroup > hybridThreshold {
        return groupNormGLUFallback(x, weight: weight, bias: bias, numGroups: numGroups, eps: eps)
    }

    let xContig = contiguous(x.reshaped([b, cFull, spatialSize]))
    let wF32 = contiguous(weight.asType(.float32))
    let bF32 = contiguous(bias.asType(.float32))
    let epsArr = MLXArray([eps])
    let params = MLXArray([Int32(numGroups), Int32(channelsPerGroup), Int32(spatialSize), Int32(cFull), Int32(cHalf)])

    let totalGroups = b * numGroups
    let tg = min(1024, max(32, ((elemsPerGroup + 31) / 32) * 32))

    var outShape = origShape
    outShape[1] = cHalf

    let result = getGroupNormGLUKernel()(
        [xContig, wF32, bF32, epsArr, params],
        template: [("T", x.dtype)],
        grid: (totalGroups * tg, 1, 1),
        threadGroup: (tg, 1, 1),
        outputShapes: [[b, cHalf, spatialSize]],
        outputDTypes: [x.dtype]
    )[0]

    return result.reshaped(outShape)
}
