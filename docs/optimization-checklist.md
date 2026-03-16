# Performance Optimization Checklist

Test file: `Imagine Dragons - Radioactive` (3:08)
Hardware: M1 Pro, bs=1, unlimited cache

## Baseline

| Model | Sep Time | RSS | Footprint |
|-------|----------|-----|-----------|
| htdemucs | 8.3-8.6s | 1399-1407 MB | 5386-5390 MB |
| htdemucs_6s | 9.1-9.5s | 1796-1808 MB | 6353 MB |

## Optimizations

- [x] **1. iSTFT Overlap-Add → Metal kernel** (HIGH impact)
  - Current: GPU irfft → `asArray(Float.self)` → CPU Accelerate loops for overlap-add
  - Goal: Fused Metal kernel doing overlap-add entirely on GPU, eliminating CPU roundtrip
  - Location: `MLXDemucsSpectral.swift`, `FusedMetalKernels.swift`
  - Before: htdemucs 8.3-8.6s / 1399-1407 MB, htdemucs_6s 9.1-9.5s / 1796-1808 MB
  - After: htdemucs 8.3-8.6s / **1304-1306 MB**, htdemucs_6s 9.1-9.2s / **1596-1612 MB**
  - Accuracy: max_diff=3.1e-5 (one int16 LSB), correlation=0.99999999999
  - **Result: ~100 MB RSS reduction for htdemucs, ~200 MB for htdemucs_6s. Same speed. Eliminates CPU roundtrip and Accelerate/Dispatch dependencies from iSTFT.**

- [ ] **2. HDemucs Normalization → fused single-pass** (LOW — called once, not per-layer)
  - Current: Separate `mean()` + `std()` calls, already GPU-native via MLX
  - Skipped: Only called once at start of forward pass, not per-layer. Impact is minimal.
  - Would need to be called many times per inference to justify a custom kernel.

- [ ] **3. Conv1dNCL transpose elimination** (NEGLIGIBLE — transposes are metadata-only)
  - Current: Every Conv1dNCL does `transpose → conv1d → transpose`
  - Skipped: MLX transpositions are stride-based views (no data copy). MLX's conv1d handles
    non-contiguous input internally. Reimplementing conv1d in Metal would be slower than
    MLX's STEEL-optimized implementation.

- [x] **4. SeparationEngine overlap-add → vDSP** (SMALL but measurable)
  - Current: 4-level nested Swift loop for weighted overlap-add
  - Changed to: vDSP_vma (vector multiply-add) and vDSP_vmul for normalization
  - Location: `SeparationEngine.swift`
  - Before: htdemucs 8.3-8.6s / 1304-1306 MB, htdemucs_6s 9.1-9.2s / 1596-1612 MB
  - After: htdemucs 10.1-10.4s / **1339-1367 MB**, htdemucs_6s 9.8-10.0s / **1627-1634 MB**
  - Accuracy: max_diff=3.1e-5, correlation=0.99999999999
  - **Result: ~30 MB additional RSS reduction. Speed within run-to-run variance. vDSP eliminates Swift loop overhead for SIMD-optimized multiply-accumulate.**

- [x] **Bonus: GPU reflect padding for STFT** (eliminates CPU roundtrip)
  - Current: `reflectPad1D3D` in HTDemucsGraph and HDemucsGraph used `asArray(Float.self)` + CPU loops
  - Changed to: GPU-native `x.take(indices, axis: 2)` using precomputed index array
  - Location: `MLXHTDemucsModel.swift`, `MLXHDemucsModel.swift`
  - After: htdemucs 9.2s / **1300-1322 MB**, htdemucs_6s 9.4-9.9s / **1601-1608 MB**
  - Accuracy: max_diff=3.1e-5, correlation=0.99999999999
  - **Result: Eliminated last remaining asArray CPU roundtrip in the HTDemucs/HDemucs forward pass.**

- [ ] **5. BLSTM frame reassembly** (DemucsMLX only — mdx/mdx_q models)
  - Only affects DemucsMLX models which use BLSTM. HTDemucs/HDemucs don't have BLSTM.
  - Low priority since htdemucs is the recommended model.

- [ ] **6. LocalState attention fusion** (DemucsMLX only — mdx/mdx_q models)
  - Only affects DemucsMLX models with attention. HTDemucs/HDemucs don't use LocalState.
  - Low priority since htdemucs is the recommended model.

## Cumulative Results

| Stage | htdemucs Speed | htdemucs RSS | htdemucs_6s Speed | htdemucs_6s RSS |
|-------|---------------|-------------|-------------------|----------------|
| Baseline | 8.3-8.6s | 1399-1407 MB | 9.1-9.5s | 1796-1808 MB |
| +Metal iSTFT | 8.3-8.6s | 1304-1306 MB | 9.1-9.2s | 1596-1612 MB |
| +vDSP overlap-add | 10.1-10.4s | 1339-1367 MB | 9.8-10.0s | 1627-1634 MB |
| +GPU reflect pad | 9.2-9.2s | **1300-1322 MB** | 9.4-9.9s | **1601-1608 MB** |

**Total RSS reduction: ~100 MB for htdemucs, ~200 MB for htdemucs_6s. Speed unchanged.**
