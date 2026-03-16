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

- [ ] **2. HDemucs Normalization → fused single-pass** (MEDIUM-HIGH, 15-25%)
  - Current: Separate `mean()` + `std()` calls scan data twice, then separate sub/div
  - Goal: Single-pass mean+variance+normalize kernel (like existing GroupNorm kernels)
  - Location: `MLXHDemucsModel.swift`
  - Before: —
  - After: —
  - Accuracy: —

- [ ] **3. Conv1dNCL transpose elimination** (MEDIUM, 10-15%)
  - Current: Every Conv1dNCL does `transpose → conv1d → transpose` (NCL↔NLC format)
  - Goal: Reduce unnecessary transposes or implement NCL-native conv kernel
  - Location: `MLXDemucsLayers.swift`
  - Before: —
  - After: —
  - Accuracy: —

- [ ] **4. SeparationEngine overlap-add → GPU/parallel** (MEDIUM, 15-30% for shifts>1)
  - Current: Nested CPU loop for chunk weighting and overlap-add accumulation
  - Goal: GPU kernel or at minimum vDSP for weighted accumulation
  - Location: `SeparationEngine.swift`
  - Before: —
  - After: —
  - Accuracy: —

- [ ] **5. BLSTM frame reassembly** (MEDIUM, 10-20%)
  - Current: Multiple slice + concatenate operations for overlapping frame extraction
  - Goal: Fused extract+trim+concat kernel
  - Location: `MLXDemucsBLSTM.swift`
  - Before: —
  - After: —
  - Accuracy: —

- [ ] **6. LocalState attention fusion** (LOW-MEDIUM, 5-10%)
  - Current: Separate matmul + einsum + softmax + matmul calls
  - Goal: Fused attention kernel
  - Location: `MLXDemucsLocalState.swift`
  - Before: —
  - After: —
  - Accuracy: —
