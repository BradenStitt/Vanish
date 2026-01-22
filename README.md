# Vanish

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

**Automatically detect and blur background faces in images and videos using state-of-the-art AI models.**

Vanish combines [Florence-2](https://huggingface.co/microsoft/Florence-2-large-ft) for intelligent face detection with [SAM2](https://github.com/facebookresearch/segment-anything-2) for precise segmentation, enabling privacy-preserving content creation with a single command.

## Features

- **Smart Speaker Detection** — Automatically identifies and preserves main speakers while blurring background faces
- **Precise Segmentation** — Uses SAM2 for pixel-perfect face masking (no ugly rectangles)
- **Image & Video Support** — Process single images or entire videos
- **Adjustable Pixelation** — Control blur intensity with customizable pixel size
- **GPU Accelerated** — Supports CUDA, MPS (Apple Silicon), and CPU

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BradenStitt/Vanish.git
cd Vanish

# Install dependencies
pip install -r requirements.txt

# Install SAM2 (required)
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Usage

**Command Line:**
```bash
# Process an image
python vanish.py input.jpg output.jpg

# Process a video
python vanish.py input.mp4 output.mp4

# Adjust pixelation intensity
python vanish.py input.jpg output.jpg --pixel-size 15

# Blur ALL faces (including main speakers)
python vanish.py input.jpg output.jpg --blur-all
```

**Python API:**
```python
from vanish import vanish, vanish_video

# Process image
result = vanish("input.jpg", "output.jpg", pixel_size=10)

# Process video
vanish_video("input.mp4", "output.mp4", pixel_size=10)

# Blur all faces (no speaker exclusion)
result = vanish("input.jpg", "output.jpg", exclude_speakers=False)
```

**Jupyter Notebook:**

Open `vanish.ipynb` for an interactive walkthrough with visualizations.

## How It Works

```
Input Image/Video
        │
        ▼
┌───────────────────┐
│    Florence-2     │  ← Object detection: finds all "human face" instances
│   (Detection)     │  ← Phrase grounding: identifies "main speaker" faces
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Filter Faces     │  ← Excludes main speakers from blur list
└───────────────────┘
        │
        ▼
┌───────────────────┐
│      SAM2         │  ← Generates precise segmentation masks
│  (Segmentation)   │     from bounding boxes
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Pixelation      │  ← Applies blur effect only to masked regions
└───────────────────┘
        │
        ▼
   Output Image/Video
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU recommended (also works on Apple Silicon MPS or CPU)
- ~4GB VRAM for inference

## Use Cases

- **Content Creation** — Blur bystanders in vlogs and street footage
- **Privacy Compliance** — Anonymize faces for GDPR/CCPA compliance
- **Security Footage** — Redact identities while preserving context
- **Social Media** — Protect privacy of people in background

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Florence-2](https://huggingface.co/microsoft/Florence-2-large-ft) by Microsoft
- [SAM2](https://github.com/facebookresearch/segment-anything-2) by Meta AI
