# libea-audio

Standalone pure-Rust EA `SCHl` / ASF / `EA-XA` / `EACS` audio codec and stream decoder. Zero dependencies, `#![forbid(unsafe_code)]`, `no_std` + `alloc` + `std` features, MSRV 1.70.

## Features

- Chunk framing: `SCHl`/`SCCl`/`SCDl`/`SCLl`/`SCEl` and big-endian `SCHb`/`SCDb`/`SCEb`.
- PT TLV header (`PT\x01\x00` at `+0x08` or `+0x0c` after a PS1 stream word): tags `0x82` channels, `0x83` codec, `0x84` sample rate, `0x85` num samples, `0x86`/`0x87` loop start/end, `0x8a` flags; markers `0xfc`/`0xfd`/`0xfe`/`0xff`.
- Decoders: EA-XA v1/v2 (mono & stereo interleaved), PS1 SPU-ADPCM payloads in PS1 `SCHl`, PCM8, PCM16 LE/BE, EA-ADPCM R1/R2/R3.
- Streaming API: zero-alloc `EaStreamView<'a>`, `EaDecoder::decode_i16`, `decode_f32_planar`, `reset()`, loop handling, bounded `Error`.

Each `SCDl` payload starts with an 8-byte sub-header, skipped by the decoder. PS1 SPU blocks are 16 bytes (28 samples); stereo blocks alternate L/R. EA-XA subframes are 15 bytes (28 samples per channel); stereo subframes are channel-sequential and interleaved on output.

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
