# demucs-mlx-swift

Swift package for Demucs music source separation on Apple Silicon, using the [MLX](https://github.com/ml-explore/mlx-swift) GPU framework.

Separates audio into stems: drums, bass, other, vocals (and guitar + piano for the 6-stem model).

## Features

- `DemucsMLX` library target (importable from macOS/iOS apps)
- `demucs-mlx-swift` CLI demo
- All 8 pretrained Demucs models supported (HTDemucs, HDemucs, Demucs v3)
- Chunked overlap-add inference with configurable segment/overlap/batch/shifts
- Async separation API with progress reporting, ETA estimation, and cancellation
- Multi-format output: WAV (16/24/32-bit), FLAC, ALAC, AAC
- Two-stem mode (e.g. vocals + no_vocals)
- Automatic model download from Hugging Face

## Supported Models

All 8 pretrained Demucs models are supported. Benchmarks on a 3:19 track (M1 Pro, batch_size=8):

| Model | Type | Stems | Time |
|-------|------|-------|------|
| `htdemucs` | HTDemucs (hybrid transformer) | 4 | 14.8s |
| `htdemucs_ft` | HTDemucs (fine-tuned, bag of 4) | 4 | 95.8s |
| `htdemucs_6s` | HTDemucs (6-stem) | 6 | 17.7s |
| `hdemucs_mmi` | HDemucs (hybrid, bag of 4) | 4 | 44.5s |
| `mdx` | Demucs v3 + HDemucs (bag of 4) | 4 | 140.7s |
| `mdx_extra` | HTDemucs (bag of 4) | 4 | 163.7s |
| `mdx_q` | Demucs v3 + HDemucs (bag of 4) | 4 | 120.6s |
| `mdx_extra_q` | HTDemucs (bag of 4) | 4 | 153.3s |

Model files are downloaded automatically from [Hugging Face](https://huggingface.co/iky1e/demucs-mlx) on first use.

## Requirements

- Swift 6.2+
- macOS 14+ or iOS 17+
- Xcode 15+
- Apple Silicon
- MLX Metal runtime with `mlx.metallib` available next to the executable

## Installation (SPM)

```swift
dependencies: [
    .package(url: "https://github.com/kylehowells/demucs-mlx-swift", branch: "master")
]
```

Then add product dependency and import:

```swift
import DemucsMLX
```

## Library Usage

### Synchronous

```swift
import DemucsMLX

let separator = try DemucsSeparator(modelName: "htdemucs")

let result = try separator.separate(fileAt: URL(fileURLWithPath: "song.mp3"))

for (source, audio) in result.stems {
    let url = URL(fileURLWithPath: "\(source).wav")
    try AudioIO.writeAudio(audio, to: url, format: .wav(bitDepth: .int16))
}
```

### Async with Progress and Cancellation

```swift
let separator = try DemucsSeparator(modelName: "htdemucs")
let cancelToken = DemucsCancelToken()

separator.separate(
    fileAt: inputURL,
    cancelToken: cancelToken,
    progress: { progress in
        // Called on main queue
        print("\(Int(progress.fraction * 100))% - \(progress.stage)")
        if let eta = progress.estimatedTimeRemaining {
            print("ETA: \(Int(eta))s")
        }
    },
    completion: { result in
        // Called on main queue
        switch result {
        case .success(let separation):
            for (source, audio) in separation.stems {
                try? AudioIO.writeAudio(audio, to: outputDir.appendingPathComponent("\(source).wav"))
            }
        case .failure(let error):
            print("Error: \(error)")
        }
    }
)

// To cancel:
cancelToken.cancel()
```

### Custom Parameters

```swift
let separator = try DemucsSeparator(
    modelName: "htdemucs_ft",
    parameters: DemucsSeparationParameters(
        shifts: 2,         // shift augmentations (improves quality, multiplies time)
        overlap: 0.25,     // overlap ratio between segments
        split: true,       // chunked overlap-add inference
        segmentSeconds: nil, // nil = use model default
        batchSize: 1,      // chunks processed in parallel (1 = lowest memory)
        seed: 42           // deterministic shifts
    ),
    modelDirectory: URL(fileURLWithPath: "/path/to/models")
)
```

## CLI Demo

Build:

```bash
make build
```

Run:

```bash
.build/release/demucs-mlx-swift track.mp3 -o separated
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --name` | Model name (default: `htdemucs`) |
| `-o, --out` | Output directory (default: `separated`) |
| `--model-dir` | Local model directory |
| `--segment` | Segment length in seconds |
| `--overlap` | Overlap ratio [0, 1) (default: 0.25) |
| `--shifts` | Shift augmentations (default: 1) |
| `--seed` | Random seed for deterministic shifts |
| `-b, --batch-size` | Chunk batch size (default: 1) |
| `--no-split` | Disable chunked overlap-add |
| `--two-stems` | Output one stem + complement (e.g. `vocals`) |
| `--async` | Use async API with progress reporting |
| `--list-models` | List available models |
| `--mp3` | Output as AAC in .m4a |
| `--flac` | Output as FLAC lossless |
| `--alac` | Output as Apple Lossless in .m4a |
| `--int24` | Output 24-bit integer WAV |
| `--float32` | Output 32-bit float WAV |

Examples:

```bash
# Separate vocals only
.build/release/demucs-mlx-swift song.mp3 --two-stems vocals -o out

# Use fine-tuned model with FLAC output
.build/release/demucs-mlx-swift song.mp3 -n htdemucs_ft --flac -o out

# 6-stem separation (drums, bass, other, vocals, guitar, piano)
.build/release/demucs-mlx-swift song.mp3 -n htdemucs_6s -o out

# Async with progress bar
.build/release/demucs-mlx-swift song.mp3 --async -o out
```

## Performance Tuning

The default `batchSize=1` provides the best balance of speed and memory for most models. Larger batch sizes increase memory usage and are often slower due to exceeding GPU cache limits.

Benchmarks on a 3:13 track (M1 Pro):

| Model | bs=1 | bs=4 | bs=8 |
|-------|------|------|------|
| `htdemucs` | 9.3s / 1.2GB | **8.5s** / 1.8GB | 10.3s / 2.1GB |
| `htdemucs_6s` | **9.3s / 1.5GB** | 10.3s / 2.5GB | 45.1s / 2.1GB |
| `hdemucs_mmi` | **9.2s / 2.1GB** | 19.6s / 3.2GB | 32.2s / 3.6GB |

For memory-constrained environments (iOS apps), also consider limiting MLX's memory cache:

```swift
import MLX

// Reduce MLX cache to prevent holding onto large GPU allocations after use.
// Default is unlimited; 2 MB keeps peak RSS under ~1.3 GB with no speed penalty
// for single-model inference. Larger caches (unlimited) can give ~25% speedup
// on some workloads at the cost of ~5x memory retention.
Memory.cacheLimit = 2 * 1024 * 1024
```

## Model Resolution

Models are resolved in this order:

1. Explicit `--model-dir` (or library `modelDirectory` parameter)
2. `DEMUCS_MLX_SWIFT_MODEL_DIR` environment variable
3. `~/.cache/demucs-mlx-swift-models/<model>`
4. Local paths: `.scratch/models/<model>`, `Models/<model>`, `./<model>`
5. Hugging Face download (default repo: `iky1e/demucs-mlx`)

Environment overrides:

- `DEMUCS_MLX_SWIFT_MODEL_REPO` — set to a Hub repo ID (`org/repo`) or URL.

## Metal Shader Library (Required for MLX Inference)

DemucsMLX treats MLX/Metal as a required runtime, matching the Swift MLX example projects. The library uses MLX GPU operations and repo-owned MLXFast custom kernels as part of the normal inference path; it does not silently fall back to CPU execution for missing or debugged Metal.

Build with `make build` or `make debug` so SwiftPM and the MLX Metal shader library are produced together. If you run `swift build` manually, generate the shader library afterward:

```bash
./scripts/build_mlx_metallib.sh release
```

If `mlx.metallib` is missing at runtime, MLX inference will fail with an error such as `Failed to load the default metallib`. If you run an `xcodebuild`/DerivedData binary, place `mlx.metallib` next to that executable.

For Metal capture/debugging, use the MLX debugging setup (`MLX_METAL_DEBUG` build plus `MTL_CAPTURE_ENABLED=1`). Custom Demucs kernels are expected to run under Metal tooling; issues there should be debugged as kernel or MLX integration issues rather than bypassed automatically.

If you see `missing Metal Toolchain`, run:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Make Targets

```bash
make build   # release + mlx.metallib
make debug   # debug + mlx.metallib
make test
make clean
```

## Exporting Models from PyTorch

A script is included to export all 8 pretrained models directly from the original PyTorch Demucs package:

```bash
pip install demucs safetensors numpy
python scripts/export_from_pytorch.py --out-dir ~/.cache/demucs-mlx-swift-models
```
