# LTX joint training prompts

Prepared OmniNFT VGGSound prompt data for the LTX2.3 joint training recipe.
These files contain text prompts and metadata, not audio/video media.

| File | Rows |
| --- | ---: |
| `train.parquet` | 19,487 |
| `test.parquet` | 16 |

Source: [OmniNFT VGGSound metadata](https://github.com/zghhui/OmniNFT/tree/fb9237f6e74edf0d0f2a683f4d975b79fde588fe/dataset/vggsound),
revision `fb9237f6e74edf0d0f2a683f4d975b79fde588fe`.
Converted with `examples/unigrpo_nft_trainer/ltx2/prepare_omninft_data.py`.
`prompt_av` supplies the generation prompt; `prompt_v` and `prompt_a` supply
video/audio reward targets. Source metadata is retained in `extra_info`.

After cloning this branch, no dataset conversion is needed. Set
`LTX_DATA_HOST` to this directory's absolute host path in the server config.

Verify downloaded files from this directory:

```bash
sha256sum -c SHA256SUMS
```
