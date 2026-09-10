# OmniNFT VGGSound prompt metadata

This directory contains the training metadata shipped by the native OmniNFT
repository [`zghhui/OmniNFT`](https://github.com/zghhui/OmniNFT), commit
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe`.

`train_metadata_20k.jsonl` has 19,487 prompt-group records. It contains text
metadata only; video and audio candidates are generated online by the old
policy. The SHA-256 of the bundled file is:

```text
e2b3629e4aa729bb91b780c5933f49e2044e603f38ea3c730e5ee99313d3db66
```

Each record has `prompt_v`, `prompt_a`, `prompt_av`, `idx`, and `category`.

Validation JSONL may contain only `prompt_v`, `prompt_a`, and `prompt_av`.
The preparation script assigns those records deterministic zero-based indices
and the synthetic category `validation`; the original metadata remains
available under `extra_info.metadata` in the converted parquet.

Convert the native files into the standard `RLHFDataset` parquet schema with:

```bash
python3 examples/omnift_trainer/ltx2/prepare_data.py \
  --train_file /path/to/train_metadata_20k.jsonl \
  --val_file /path/to/test_metadata.jsonl \
  --output_dir ./data/omninft/vggsound/verl_omni
```

See [`examples/omnift_trainer/README.md`](../../../examples/omnift_trainer/README.md)
for the output schema and launcher configuration.
