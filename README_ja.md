# Zonos Gradio 5 パッチ

[Zyphra/Zonos](https://github.com/Zyphra/Zonos) を Gradio 5 環境で安定動作させるためのパッチファイルです。

SkyrimNet、CHIM などの日本語TTSクライアントでの利用を想定しています。

**[English README](README.md)**

## 概要

このパッチは、オリジナルの `gradio_interface.py` に以下の機能を追加・修正したものです：

| 機能 | 説明 |
|------|------|
| Gradio 5 互換パッチ | ファイルパス処理のエラーを回避 |
| テキストサニタイズ | eSpeak非対応の特殊文字を除去・変換 |
| テキスト長制限 | 100文字超過時に切り取り（タイムアウト防止） |
| pingリクエスト対応 | ヘルスチェック用の無音応答 |
| バージョン表示 | 起動時にパッチバージョンを表示 |
| デバッグログ | 処理テキストの可視化 |

## 動作環境

- **Zonos**: v0.1
- **Gradio**: 5.x
- **OS**: Ubuntu (WSL2推奨)
- **GPU**: NVIDIA 6GB+ VRAM

## インストール方法

1. WSL (Ubuntu) 上で [Zyphra/Zonos](https://github.com/Zyphra/Zonos) を通常通りインストール
2. このリポジトリから `gradio_interface.py` をダウンロード
3. WSL上のZonosインストールディレクトリにある `gradio_interface.py` を、ダウンロードしたファイルで上書き
4. WSL上でZonosを起動

起動時に以下のようなバナーが表示されれば成功です：

```
==================================================
  Zonos Server Patch Version: 1.0.4
==================================================
```

## 変更点の詳細

### 1. Gradio 5 互換パッチ

Gradio 5のファイルパス処理で発生するエラーを回避するために、以下の関数をラップしています：

- `gradio.processing_utils._check_allowed`
- `gradio.processing_utils.hash_file`
- `gradio.processing_utils.save_file_to_cache`
- `gradio.blocks.Block.async_move_resource_to_block_cache`

> ⚠️ **注意**: セキュリティチェックを一部バイパスしています。信頼できるネットワーク環境でのみ使用してください。

### 2. テキストサニタイズ

eSpeak（phonemizer）が処理できない文字を変換・除去します：

- 全角英数字 → 半角変換
- 特殊記号（★、♪、【】など）→ 除去

### 3. テキスト長制限

`MAX_TEXT_LENGTH = 100` で設定されています。超過分は切り取られ、ログに表示されます：

```
[WARN] Text truncated: 250 -> 100 chars
[TEXT] こんにちは...(100文字)
[TEXT_TRUNCATED] ...(切り取られた残り)
```

### 4. その他の調整

- `prefix_audio`: 現在無効化（安定性のため）
- `emotion`: unconditional設定を強制（自動判定に任せる）


## 設定変更

`gradio_interface.py` 内の定数を編集することでカスタマイズできます：

```python
MAX_TEXT_LENGTH = 100  # テキスト長制限（増減可能）
```

## ライセンス

Apache License 2.0

Based on [Zyphra/Zonos](https://github.com/Zyphra/Zonos)

## 謝辞

- [Zyphra](https://www.zyphra.com/) - Zonos TTS
- SkyrimNet / CHIM コミュニティ
