# libea-audio

Standalone pure-Rust EA `SCHl` / ASF / `EA-XA` audio codec and stream decoder. Zero dependencies, `#![forbid(unsafe_code)]`, `no_std` + `alloc` + `std` features, MSRV 1.70.

## Features

- Chunk framing: `SCHl`/`SCCl`/`SCDl`/`SCLl`/`SCEl` and big-endian `SCHb`/`SCDb`/`SCEb`.
- PT TLV header (`PT\x01\x00`, `PT\x00\x00`, or `GSTR` at `+0x08`, or `PT` at `+0x0c` after a PS1 stream word): tags `0x80` version/platform, `0x81` sample width, `0x82` channels, `0x83` codec (`0x00` PCM, `0x04` MT10, `0x05` EA-XA v1 / PS1 SPU, `0x07` EA-XA v2 / EA-ADPCM R1-R3, `0x16` EA-XAS), `0x84` sample rate, `0x85` num samples, `0x86`/`0x87` loop start/end, `0x8a` flags; extended `0xFF` 4-byte TLV lengths; markers `0xfc`/`0xfd`/`0xfe`/`0xff`.
- Decoders: EA-XA v1 (mono 15-byte & stereo 30-byte nibble-interleaved blocks), EA-XA v2 (15-byte mono subframes), PS1 SPU-ADPCM payloads in PS1 `SCHl`, PCM8, PCM16 LE/BE, EA-ADPCM R1/R2/R3 (with per-chunk `(prev1, prev2)` initial history).
- Streaming API: zero-alloc `EaStreamView<'a>`, `EaDecoder::decode_i16`, `decode_f32_planar`, `reset()`, O(1) cached loop wrapping, bounded `Error`.

`SCDl`/`SCDb` subheader lengths are codec/generation-aware: 4-byte sample count for PCM and EA-XA v1/v2, 8-byte subheader (4-byte PS1 stream word + 4-byte sample count) for PS1 `SCHl` / SPU-ADPCM, and `4 + 4 * channels` bytes (8B mono / 12B stereo) with per-channel `(prev1, prev2)` `i16` initial history for EA-ADPCM R1/R2/R3.

## Usage

```rust
use libea_audio::{EaDecoder, EaStreamView};

fn main() -> libea_audio::Result<()> {
    let bytes: &[u8] = &[]; // load an SCHl/ASF stream
    let view = EaStreamView::parse(bytes)?;
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 1024];
    let frames = dec.decode_i16(&mut out)?;
    let _ = frames;
    Ok(())
}
```

Dual-licensed under MIT OR Apache-2.0.
