// SPDX-License-Identifier: MIT OR Apache-2.0
//! EA audio decoding throughput benchmark (`harness = false`).

use std::hint::black_box;
use std::time::Instant;

use libea_audio::{EaDecoder, EaStreamView};

fn chunk(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(tag);
    o.extend_from_slice(&((payload.len() + 8) as u32).to_le_bytes());
    o.extend_from_slice(payload);
    o
}

fn build_stream(stereo_spu: bool) -> Vec<u8> {
    let channels: u8 = if stereo_spu { 2 } else { 1 };
    let mut pt = Vec::new();
    pt.extend_from_slice(&0x1234u32.to_le_bytes());
    pt.extend_from_slice(b"PT\x01\x00");
    pt.push(0xFD);
    pt.extend_from_slice(&[0x82, 1, channels]);
    // No codec tag -> PS1 SPU inference (ps1 word present). For XA bench
    // we reuse the same stream; SPU is the hot PS1 path.
    pt.extend_from_slice(&[0x84, 2, 0x56, 0x22]);
    pt.extend_from_slice(&[0x8A, 1, 0, 0xFF]);
    let mut s = chunk(b"SCHl", &pt);
    // 200 SCDl chunks, each 8-byte header + 100 SPU blocks (2800 samples/ch worth of blocks)
    for _ in 0..200 {
        let mut p = vec![0u8; 8];
        for _ in 0..100 {
            p.push(0x29);
            p.push(0x02);
            p.extend_from_slice(&[0x47; 14]);
            if stereo_spu {
                p.push(0x12);
                p.push(0x02);
                p.extend_from_slice(&[0x47; 14]);
            }
        }
        s.extend_from_slice(&chunk(b"SCDl", &p));
    }
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    s
}

fn bench_one(name: &str, stream: &[u8]) {
    let view = EaStreamView::parse(stream).unwrap();
    let channels = view.header().channels as usize;
    let mut dec = EaDecoder::new(view);
    let mut out = vec![0i16; 8192 * channels];
    // Warm up + measure over full passes.
    let passes = 20;
    let t0 = Instant::now();
    let mut total_frames: u64 = 0;
    for _ in 0..passes {
        dec.reset();
        loop {
            let n = dec.decode_i16(black_box(&mut out)).unwrap();
            if n == 0 {
                break;
            }
            total_frames += n as u64;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let msamples = (total_frames as f64 / 1_000_000.0) / elapsed;
    println!(
        "{name}: {msamples:.2} MSamples/sec ({total_frames} frames in {elapsed:.3}s, {} passes)",
        passes
    );
}

fn main() {
    let mono = build_stream(false);
    let stereo = build_stream(true);
    bench_one("ps1-spu-mono  ", &mono);
    bench_one("ps1-spu-stereo", &stereo);
}
