# Zonos Gradio 5 Patch

A patch file for stable operation of [Zyphra/Zonos](https://github.com/Zyphra/Zonos) in Gradio 5 environments.

Designed for Japanese TTS clients such as SkyrimNet and CHIM.

**[日本語版 README はこちら](README_ja.md)**

## Overview

This patch adds the following features and fixes to the original `gradio_interface.py`:

| Feature | Description |
|---------|-------------|
| Gradio 5 Compatibility | Fixes file path handling errors |
| Text Sanitization | Removes/converts special characters unsupported by eSpeak |
| Text Length Limit | Truncates text over 100 characters (prevents timeout) |
| Ping Request Support | Silent audio response for health checks |
| Version Display | Shows patch version at startup |
| Debug Logging | Visualizes processed text |

## Requirements

- **Zonos**: v0.1
- **Gradio**: 5.x
- **OS**: Ubuntu (WSL2 recommended)
- **GPU**: NVIDIA 6GB+ VRAM

## Installation

1. Install [Zyphra/Zonos](https://github.com/Zyphra/Zonos) as usual
2. Replace `gradio_interface.py` with this patched version:

```bash
cp gradio_interface.py ~/Zonos/gradio_interface.py
```

3. Start Zonos:

```bash
cd ~/Zonos
source .venv/bin/activate
uv run gradio_interface.py
```

If successful, you will see the following banner at startup:

```
==================================================
  Zonos Server Patch Version: 1.0.3
==================================================
```

## Changes in Detail

### 1. Gradio 5 Compatibility Patch

Wraps the following functions to avoid file path handling errors in Gradio 5:

- `gradio.processing_utils._check_allowed`
- `gradio.processing_utils.hash_file`
- `gradio.processing_utils.save_file_to_cache`
- `gradio.blocks.Block.async_move_resource_to_block_cache`

> ⚠️ **Warning**: Some security checks are bypassed. Use only in trusted network environments.

### 2. Text Sanitization

Converts or removes characters that eSpeak (phonemizer) cannot process:

- Full-width alphanumeric → Half-width conversion
- Special symbols (★, ♪, 【】, etc.) → Removed

### 3. Text Length Limit

Set to `MAX_TEXT_LENGTH = 100`. Excess text is truncated and logged:

```
[WARN] Text truncated: 250 -> 100 chars
[TEXT] Hello...(100 chars)
[TEXT_TRUNCATED] ...(remaining truncated text)
```

### 4. Other Adjustments

- `prefix_audio`: Currently disabled (for stability)
- `emotion`: Forced to unconditional (auto-detection)
- `seed`: Fixed at 420 (for reproducibility, can be changed)

## Configuration

You can customize by editing constants in `gradio_interface.py`:

```python
MAX_TEXT_LENGTH = 100  # Text length limit (adjustable)
```

## License

Apache License 2.0

Based on [Zyphra/Zonos](https://github.com/Zyphra/Zonos)

## Acknowledgments

- [Zyphra](https://www.zyphra.com/) - Zonos TTS
- SkyrimNet / CHIM Community
