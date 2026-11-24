//! Clean-room synthetic integration tests (no copyrighted bytes).

use libea_audio::{ChunkKind, Codec, EaDecoder, EaStreamView};

fn chunk(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(tag);
    o.extend_from_slice(&((payload.len() + 8) as u32).to_le_bytes());
    o.extend_from_slice(payload);
    o
}
fn chunk_be(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(tag);
    o.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
    o.extend_from_slice(payload);
    o
}

fn pt(channels: u32, codec: Option<u32>, rate: u32, num: u32, ps1: bool, extra: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    if ps1 {
        p.extend_from_slice(&0x1234u32.to_le_bytes());
    }
    p.extend_from_slice(b"PT\x01\x00");
    p.push(0xFD);
    p.extend_from_slice(&[0x82, 1, channels as u8]);
    if let Some(c) = codec {
        p.extend_from_slice(&[0x83, 1, c as u8]);
    }
    p.extend_from_slice(&[0x84, 2, (rate >> 8) as u8, rate as u8]);
    p.extend_from_slice(&[
        0x85,
        4,
        (num >> 24) as u8,
        (num >> 16) as u8,
        (num >> 8) as u8,
        num as u8,
    ]);
    p.extend_from_slice(extra);
    p.extend_from_slice(&[0x8A, 1, 0]);
    p.push(0xFF);
    p
}

fn stream(codec: Option<u32>, channels: u32, data_payloads: &[Vec<u8>], ps1: bool) -> Vec<u8> {
    let total: u32 = data_payloads
        .iter()
        .map(|d| (d.len() / 16) as u32 * 28)
        .sum();
    let mut s = chunk(b"SCHl", &pt(channels, codec, 22050, total, ps1, &[]));
    s.extend_from_slice(&chunk(b"SCCl", &1u32.to_le_bytes()));
    for d in data_payloads {
        let mut p = vec![0u8; 8];
        p.extend_from_slice(d);
        s.extend_from_slice(&chunk(b"SCDl", &p));
    }
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    s
}

fn spu_body(blocks: usize, hdr: u8) -> Vec<u8> {
    let mut b = Vec::new();
    for _ in 0..blocks {
        b.push(hdr);
        b.push(0x02);
        b.extend_from_slice(&[0x11; 14]);
    }
    b
}

#[test]
fn ps1_mono_spu_decodes_exact_frame_count() {
    // 4 SPU blocks = 112 samples.
    let s = stream(None, 1, &[spu_body(4, 0x29)], true);
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().codec, Codec::Ps1SpuAdpcm);
    assert_eq!(view.header().ps1_stream_word, Some(0x1234));
    assert_eq!(view.header().num_samples, Some(112));
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 256];
    let mut total = 0;
    loop {
        let n = dec.decode_i16(&mut out).unwrap();
        if n == 0 {
            break;
        }
        total += n;
    }
    assert_eq!(total, 112);
    assert!(dec.is_finished());
}

#[test]
fn ps1_stereo_spu_interleaved() {
    // 4 block pairs = 112 frames stereo.
    let mut body = Vec::new();
    for _ in 0..4 {
        body.extend_from_slice(&spu_body(1, 0x29));
        body.extend_from_slice(&spu_body(1, 0x12));
    }
    let s = stream(None, 2, &[body], true);
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().channels, 2);
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 512];
    let n = dec.decode_i16(&mut out).unwrap();
    assert_eq!(n, 112);
}

#[test]
fn ea_xa_mono_and_stereo() {
    for channels in [1u32, 2] {
        let sub = {
            let mut v = vec![0x29u8];
            v.extend_from_slice(&[0x22; 14]);
            v
        };
        let body = if channels == 1 {
            sub.clone()
        } else {
            [sub.clone(), sub].concat()
        };
        let mut s = chunk(b"SCHl", &pt(channels, Some(5), 22050, 28, false, &[]));
        let mut p = vec![0u8; 8];
        p.extend_from_slice(&body);
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));
        let view = EaStreamView::parse(&s).unwrap();
        assert_eq!(view.header().codec, Codec::EaXaV2);
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 128];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 28);
        // f32 planar
        dec.reset();
        let mut l = [0f32; 64];
        let mut r = [0f32; 64];
        assert_eq!(dec.decode_f32_planar(&mut l, &mut r).unwrap(), 28);
        assert!(l.iter().all(|v| *v >= -1.0 && *v <= 1.0));
    }
}

#[test]
fn pcm16le_and_pcm8_and_be() {
    for (codec_id, payload, expect) in [
        (2u32, vec![0x00, 0x40, 0x00, 0xC0], vec![16384i16, -16384]),
        (3u32, vec![0x40, 0x00, 0xC0, 0x00], vec![16384i16, -16384]),
    ] {
        let mut s = chunk(b"SCHl", &pt(1, Some(codec_id), 22050, 2, false, &[]));
        let mut p = vec![0u8; 8];
        p.extend_from_slice(&payload);
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));
        let view = EaStreamView::parse(&s).unwrap();
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 16];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 2);
        assert_eq!(&out[..2], &expect[..]);
    }
    // PCM8
    let mut s = chunk(b"SCHl", &pt(1, Some(1), 22050, 2, false, &[]));
    let mut p = vec![0u8; 8];
    p.extend_from_slice(&[0x40, 0xC0]);
    s.extend_from_slice(&chunk(b"SCDl", &p));
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 16];
    assert_eq!(dec.decode_i16(&mut out).unwrap(), 2);
    assert_eq!(out[0], 0x40 << 8);
}

#[test]
fn ea_adpcm_r1_r2_r3_decode() {
    for (id, sub_len) in [(7u32, 15usize), (8, 15), (9, 16)] {
        let mut sub = vec![0x29u8];
        if sub_len == 16 {
            sub.push(0);
        }
        sub.extend_from_slice(&[0x22; 14]);
        let mut s = chunk(b"SCHl", &pt(1, Some(id), 22050, 28, false, &[]));
        let mut p = vec![0u8; 8];
        p.extend_from_slice(&sub);
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));
        let view = EaStreamView::parse(&s).unwrap();
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 64];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 28, "codec id {id}");
    }
}

#[test]
fn loop_handling_wraps() {
    // Loop 0..28 over 56 frames of SPU data: decoding 56 frames should wrap once.
    let extra = [0x86, 1, 0, 0x87, 1, 28];
    let mut hdr = pt(1, None, 22050, 56, true, &extra);
    // pt() appends flags+FF after extra, loop tags are before that: valid TLV order.
    let _ = &mut hdr;
    let mut s = chunk(b"SCHl", &hdr);
    let mut p = vec![0u8; 8];
    p.extend_from_slice(&spu_body(2, 0x29));
    s.extend_from_slice(&chunk(b"SCDl", &p));
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().loop_start, Some(0));
    assert_eq!(view.header().loop_end, Some(28));
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 128];
    let n = dec.decode_i16(&mut out).unwrap();
    // Looping streams never EOF: the 128-frame buffer is filled by wraps.
    assert_eq!(n, 128);
    assert!(dec.loop_count() >= 1);
}

#[test]
fn big_endian_chunks() {
    let mut s = chunk_be(b"SCHb", &pt(1, Some(2), 22050, 1, false, &[]));
    let mut p = vec![0u8; 8];
    p.extend_from_slice(&[0x10, 0x00]);
    s.extend_from_slice(&chunk_be(b"SCDb", &p));
    s.extend_from_slice(&chunk_be(b"SCEb", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    assert!(view.is_big_endian());
    let kinds: Vec<_> = view.chunks().map(|c| c.unwrap().kind).collect();
    assert_eq!(kinds[1], ChunkKind::Data);
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 8];
    assert_eq!(dec.decode_i16(&mut out).unwrap(), 1);
    assert_eq!(out[0], 16);
}

#[test]
fn error_cases() {
    assert!(EaStreamView::parse(b"XXXX").is_err());
    assert!(EaStreamView::parse(b"SCHl\x00\x00").is_err());
    let s = stream(Some(99), 1, &[spu_body(1, 0x29)], false);
    assert!(EaStreamView::parse(&s).is_err()); // unsupported codec
                                               // Invalid channels
    let mut s2 = chunk(b"SCHl", &pt(9, Some(2), 22050, 0, false, &[]));
    s2.extend_from_slice(&chunk(b"SCEl", &[]));
    assert!(EaStreamView::parse(&s2).is_err());
}

#[test]
fn reset_redecodes_identically() {
    let s = stream(None, 1, &[spu_body(2, 0x29)], true);
    let view = EaStreamView::parse(&s).unwrap();
    let mut dec = EaDecoder::new(view);
    let mut a = [0i16; 128];
    let na = dec.decode_i16(&mut a).unwrap();
    dec.reset();
    let mut b = [0i16; 128];
    let nb = dec.decode_i16(&mut b).unwrap();
    assert_eq!(na, nb);
    assert_eq!(a, b);
}

#[test]
fn decoder_is_send_sync() {
    fn assert_ss<T: Send + Sync>() {}
    assert_ss::<EaDecoder<'static>>();
    assert_ss::<EaStreamView<'static>>();
}
