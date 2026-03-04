# MLX Perception Encoder

MLX Swift implementation of Meta's [Perception Encoder](https://github.com/facebookresearch/perception_models) for on-device image and video search on Apple Silicon. Currently, only the PE Core checkpoints are supported.

## Supported Models

| Variant | Checkpoint | Patch | Image Size | Vision Width | Layers | Output Dim |
|---------|-----------|-------|------------|-------------|--------|------------|
| Tiny | `PE-Core-T16-384` | 16 | 384 | 192 | 12 | 512 |
| Small | `PE-Core-S16-384` | 16 | 384 | 384 | 12 | 512 |
| Base | `PE-Core-B16-224` | 16 | 224 | 768 | 12 | 1024 |
| Large | `PE-Core-L14-336` | 14 | 336 | 1024 | 24 | 1024 |
| Giant | `PE-Core-G14-448` | 14 | 448 | 1536 | 50 | 1280 |

See the upstream [PE benchmarks](https://github.com/facebookresearch/perception_models#vision-language-benchmarks) for accuracy details.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vincentamato/mlx-swift-pe", from: "0.1.0")
]
```

Then add `MLXPE` to your target:

```swift
.target(name: "YourTarget", dependencies: [
    .product(name: "MLXPE", package: "mlx-swift-pe")
])
```

## Getting Started

### 1. Convert a model

Download and convert a PE Core checkpoint to the MLX format. Requires Python with `torch`, `safetensors`, and `huggingface_hub`:

```bash
python3 scripts/convert_pe_checkpoint.py facebook/PE-Core-L14-336 ./pe-core-l14-mlx
```

This produces a directory with `config.json` and `model.safetensors`.

### 2. Use in Swift

```swift
import MLXPE

// Load model
let pe = try PECore.load(from: "./pe-core-l14-mlx")

// Encode an image
let image = NSImage(contentsOfFile: "photo.jpg")!  // UIImage on iOS
let imageEmbedding = try pe.encodeImage(image)

// Encode text
let textEmbedding = try pe.encodeText("a cat sitting on a couch")

// Compare (cosine similarity — embeddings are already L2-normalized)
let similarity = (imageEmbedding * textEmbedding).sum(axis: -1)
```

## CLI Tools

Build the tools:

```bash
xcodebuild build -scheme pe-encode -destination 'platform=OS X' -derivedDataPath .build/xcode
xcodebuild build -scheme pe-search -destination 'platform=OS X' -derivedDataPath .build/xcode

# Binaries are placed in:
BIN=.build/xcode/Build/Products/Debug
```

### pe-encode

Encode video segments into embeddings:

```bash
$BIN/pe-encode \
  --input video.mp4 \
  --model ./pe-core-l14-mlx \
  --segment-duration 10 \
  --frames-per-segment 8 \
  --output ./output
```

Use `--help` to see all options and defaults.

### pe-search

Search encoded segments with a text query:

```bash
$BIN/pe-search \
  --query "a person riding a bike" \
  --embeddings ./output/embeddings.safetensors \
  --model ./pe-core-l14-mlx
```

Use `--help` to see all options and defaults.

## Development

Build:

```bash
xcodebuild build -scheme mlx-swift-pe-Package -destination 'platform=OS X'
```

Run tests:

```bash
xcodebuild test -scheme mlx-swift-pe-Package -destination 'platform=OS X'
```

### Parity Tests

The parity tests compare layer-wise outputs from the Swift implementation against the PyTorch reference. They require generating reference resources first:

```bash
# 1. Generate reference outputs (requires torch, safetensors, huggingface_hub, PIL, requests)
python3 scripts/generate_reference_outputs.py \
  --model-name PE-Core-T16-384 \
  --output Tests/MLXPETests/Resources/ReferenceOutputs

# 2. Convert the same checkpoint for the Swift model
python3 scripts/convert_pe_checkpoint.py \
  facebook/PE-Core-T16-384 \
  Tests/MLXPETests/Resources/Model

# 3. Run parity tests
xcodebuild test -scheme mlx-swift-pe-Package -destination 'platform=OS X' \
  -only-testing:MLXPETests/ParityTests
```

The tests validate patch embedding, transformer blocks, attention pooling, projection, and final similarity within 1e-4 tolerance.

## Acknowledgments

- [Meta FAIR](https://github.com/facebookresearch/perception_models) for the Perception Encoder
- [MLX Swift](https://github.com/ml-explore/mlx-swift) for the Apple Silicon ML framework

## License

[MIT](LICENSE)
