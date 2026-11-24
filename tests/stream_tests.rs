//! Clean-room synthetic integration tests (no copyrighted bytes).

use libea_audio::{ChunkKind, Codec, EaDecoder, EaStreamView, Error};

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

fn pt_with_magic(
    magic: &[u8; 4],
    channels: u32,
    codec: Option<u32>,
    rate: u32,
    num: u32,
    ps1: bool,
    extra: &[u8],
) -> Vec<u8> {
    let mut p = Vec::new();
    if ps1 {
        p.extend_from_slice(&0x1234u32.to_le_bytes());
    }
    p.extend_from_slice(magic);
    if magic == b"GSTR" {
        p.extend_from_slice(&0u32.to_le_bytes());
    }
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

fn pt(channels: u32, codec: Option<u32>, rate: u32, num: u32, ps1: bool, extra: &[u8]) -> Vec<u8> {
    pt_with_magic(b"PT\x01\x00", channels, codec, rate, num, ps1, extra)
}

fn stream_ps1(codec: Option<u32>, channels: u32, data_payloads: &[Vec<u8>]) -> Vec<u8> {
    let total: u32 = data_payloads
        .iter()
        .map(|d| (d.len() / (16 * channels as usize)) as u32 * 28)
        .sum();
    let mut s = chunk(b"SCHl", &pt(channels, codec, 22050, total, true, &[]));
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
    let s = stream_ps1(None, 1, &[spu_body(4, 0x29)]);
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().codec, Codec::Ps1SpuAdpcm);
    assert_eq!(view.header().ps1_stream_word, Some(0x1234));
    assert_eq!(view.header().num_samples, Some(112));
    assert_eq!(view.header().scdl_subheader_len(), 8);
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
    // 4 block pairs = 112 frames stereo, explicit codec 0x05 + PS1 stream word.
    let mut body = Vec::new();
    for _ in 0..4 {
        body.extend_from_slice(&spu_body(1, 0x29));
        body.extend_from_slice(&spu_body(1, 0x12));
    }
    let s = stream_ps1(Some(0x05), 2, &[body]);
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().codec, Codec::Ps1SpuAdpcm);
    assert_eq!(view.header().channels, 2);
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 512];
    let n = dec.decode_i16(&mut out).unwrap();
    assert_eq!(n, 112);
}

#[test]
fn ea_xa_v2_mono_and_stereo() {
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
        // Real PT tag 0x83 = 0x07 -> EA-XA v2; 4-byte SCDl subheader (sample count).
        let mut s = chunk(b"SCHl", &pt(channels, Some(0x07), 22050, 28, false, &[]));
        let mut p = 28u32.to_le_bytes().to_vec();
        p.extend_from_slice(&body);
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));
        let view = EaStreamView::parse(&s).unwrap();
        assert_eq!(view.header().codec, Codec::EaXaV2);
        assert_eq!(view.header().scdl_subheader_len(), 4);
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 128];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 28);
        // f32 planar directly from group buffer
        dec.reset();
        let mut l = [0f32; 64];
        let mut r = [0f32; 64];
        assert_eq!(dec.decode_f32_planar(&mut l, &mut r).unwrap(), 28);
        assert!(l.iter().all(|v| *v >= -1.0 && *v <= 1.0));
        if channels == 1 {
            assert_eq!(&l[..28], &r[..28]);
        }
    }
}

#[test]
fn ea_xa_v1_stereo_30_byte_interleaved_block() {
    // Real PT tag 0x83 = 0x05 (non-PS1) -> EA-XA v1.
    // Stereo block = 2 header bytes (L, R) + 28 L/R nibble-interleaved bytes = 30 bytes.
    let mut block = vec![0x18u8, 0x29u8];
    block.extend_from_slice(&[0x2F; 28]); // L = +2, R = -1
    let mut s = chunk(b"SCHl", &pt(2, Some(0x05), 22050, 28, false, &[]));
    let mut p = 28u32.to_le_bytes().to_vec();
    p.extend_from_slice(&block);
    s.extend_from_slice(&chunk(b"SCDl", &p));
    s.extend_from_slice(&chunk(b"SCEl", &[]));

    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().codec, Codec::EaXaV1);
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 64];
    assert_eq!(dec.decode_i16(&mut out).unwrap(), 28);
    assert!(out[0] > 0, "left channel should be positive");
    assert!(out[1] < 0, "right channel should be negative");
}

#[test]
fn pcm16le_pcm16be_and_pcm8() {
    // PCM16 LE: SCHl + 0x83=0x00 + 0x81=2 (or 16)
    let mut s = chunk(b"SCHl", &pt(1, Some(0x00), 22050, 2, false, &[0x81, 1, 2]));
    let mut p = 2u32.to_le_bytes().to_vec();
    p.extend_from_slice(&[0x00, 0x40, 0x00, 0xC0]);
    s.extend_from_slice(&chunk(b"SCDl", &p));
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().codec, Codec::Pcm16Le);
    assert_eq!(view.header().bits_per_sample, Some(2));
    let mut dec = EaDecoder::new(view);
    let mut out = [0i16; 16];
    assert_eq!(dec.decode_i16(&mut out).unwrap(), 2);
    assert_eq!(&out[..2], &[16384i16, -16384]);

    // PCM16 BE: SCHb + 0x83=0x00 + 0x81=16
    let mut s_be = chunk_be(b"SCHb", &pt(1, Some(0x00), 22050, 2, false, &[0x81, 1, 16]));
    let mut p_be = 2u32.to_be_bytes().to_vec();
    p_be.extend_from_slice(&[0x40, 0x00, 0xC0, 0x00]);
    s_be.extend_from_slice(&chunk_be(b"SCDb", &p_be));
    s_be.extend_from_slice(&chunk_be(b"SCEb", &[]));
    let view_be = EaStreamView::parse(&s_be).unwrap();
    assert_eq!(view_be.header().codec, Codec::Pcm16Be);
    assert!(view_be.is_big_endian());
    let mut dec_be = EaDecoder::new(view_be);
    assert_eq!(dec_be.decode_i16(&mut out).unwrap(), 2);
    assert_eq!(&out[..2], &[16384i16, -16384]);

    // PCM8: SCHl + 0x83=0x00 + 0x81=1
    let mut s8 = chunk(b"SCHl", &pt(1, Some(0x00), 22050, 2, false, &[0x81, 1, 1]));
    let mut p8 = 2u32.to_le_bytes().to_vec();
    p8.extend_from_slice(&[0x40, 0xC0]);
    s8.extend_from_slice(&chunk(b"SCDl", &p8));
    s8.extend_from_slice(&chunk(b"SCEl", &[]));
    let view8 = EaStreamView::parse(&s8).unwrap();
    assert_eq!(view8.header().codec, Codec::Pcm8);
    let mut dec8 = EaDecoder::new(view8);
    assert_eq!(dec8.decode_i16(&mut out).unwrap(), 2);
    assert_eq!(out[0], 0x40 << 8);
}

#[test]
fn ea_adpcm_r1_r2_r3_with_per_chunk_history() {
    for (ver, expected_codec, sub_len) in [
        (1u8, Codec::EaAdpcmR1, 15usize),
        (2u8, Codec::EaAdpcmR2, 15usize),
        (3u8, Codec::EaAdpcmR3, 16usize),
    ] {
        // Stereo R1/R2/R3: 12-byte SCDl subheader = 4B count + L(prev1, prev2) + R(prev1, prev2).
        let mut sub_l = vec![0x10u8]; // pred 1 (c1=240, c2=0), shift 20
        let mut sub_r = vec![0x10u8];
        if sub_len == 16 {
            sub_l.push(0);
            sub_r.push(0);
        }
        sub_l.extend_from_slice(&[0x00; 14]); // zero residual nibbles -> output driven purely by history!
        sub_r.extend_from_slice(&[0x00; 14]);

        let extra = [0x80, 1, ver];
        let mut s = chunk(b"SCHl", &pt(2, Some(0x07), 22050, 28, false, &extra));
        let mut p = Vec::new();
        p.extend_from_slice(&28u32.to_le_bytes());
        p.extend_from_slice(&1000i16.to_le_bytes()); // L prev1
        p.extend_from_slice(&0i16.to_le_bytes()); // L prev2
        p.extend_from_slice(&(-1000i16).to_le_bytes()); // R prev1
        p.extend_from_slice(&0i16.to_le_bytes()); // R prev2
        p.extend_from_slice(&sub_l);
        p.extend_from_slice(&sub_r);
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));

        let view = EaStreamView::parse(&s).unwrap();
        assert_eq!(view.header().codec, expected_codec);
        assert_eq!(view.header().version, Some(u32::from(ver)));
        assert_eq!(view.header().scdl_subheader_len(), 12);
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 64];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 28, "version {ver}");
        // (1000 * 240 + 128) >> 8 = 938; (-1000 * 240 + 128) >> 8 = -937
        assert_eq!(
            out[0], 938,
            "L initial history applied for {expected_codec:?}"
        );
        assert_eq!(
            out[1], -937,
            "R initial history applied for {expected_codec:?}"
        );
    }
}

#[test]
fn pt_v0_gstr_and_extended_tlv_length() {
    // PT\x00\x00 + extended 0xFF 4-byte length on an unknown tag (6 bytes long).
    let extra = [0x99, 0xFF, 0x00, 0x00, 0x00, 0x06, 1, 2, 3, 4, 5, 6];
    for magic in [b"PT\x00\x00", b"GSTR"] {
        let mut s = chunk(
            b"SCHl",
            &pt_with_magic(magic, 1, Some(0x00), 22050, 1, false, &extra),
        );
        let mut p = 1u32.to_le_bytes().to_vec();
        p.extend_from_slice(&1234i16.to_le_bytes());
        s.extend_from_slice(&chunk(b"SCDl", &p));
        s.extend_from_slice(&chunk(b"SCEl", &[]));

        let view = EaStreamView::parse(&s).unwrap();
        assert_eq!(view.header().codec, Codec::Pcm16Le);
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 4];
        assert_eq!(dec.decode_i16(&mut out).unwrap(), 1);
        assert_eq!(out[0], 1234);
    }
}

#[test]
fn mt10_and_xas_parse_metadata_and_error_on_decode() {
    for (id, expected) in [(0x04u32, Codec::Mt10), (0x16u32, Codec::EaXas)] {
        let mut s = chunk(b"SCHl", &pt(1, Some(id), 22050, 28, false, &[]));
        s.extend_from_slice(&chunk(b"SCDl", &[0u8; 20]));
        s.extend_from_slice(&chunk(b"SCEl", &[]));
        let view = EaStreamView::parse(&s).unwrap();
        assert_eq!(view.header().codec, expected);
        let mut dec = EaDecoder::new(view);
        let mut out = [0i16; 32];
        assert_eq!(dec.decode_i16(&mut out), Err(Error::UnsupportedCodec(id)));
    }
}

#[test]
fn loop_handling_wraps_with_nonzero_loop_start_snapshot() {
    // Loop 10..28 over 56 frames of SPU data: verify exact sample repetition across wraps.
    let extra = [0x86, 1, 10, 0x87, 1, 28];
    let hdr = pt(1, None, 22050, 56, true, &extra);
    let mut s = chunk(b"SCHl", &hdr);
    let mut p = vec![0u8; 8];
    p.extend_from_slice(&spu_body(2, 0x29));
    s.extend_from_slice(&chunk(b"SCDl", &p));
    s.extend_from_slice(&chunk(b"SCEl", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    assert_eq!(view.header().loop_start, Some(10));
    assert_eq!(view.header().loop_end, Some(28));
    let mut dec = EaDecoder::new(view);
    // 10 intro + 18 loop + 18 loop + 18 loop = 64 frames
    let mut out = [0i16; 64];
    let n = dec.decode_i16(&mut out).unwrap();
    assert_eq!(n, 64);
    assert_eq!(dec.loop_count(), 2);
    // All three loop iterations (10..28, 28..46, 46..64) must match bit-for-bit.
    assert_eq!(&out[10..28], &out[28..46]);
    assert_eq!(&out[10..28], &out[46..64]);
}

#[test]
fn big_endian_chunks() {
    let mut s = chunk_be(b"SCHb", &pt(1, Some(0x00), 22050, 1, false, &[]));
    let mut p = 1u32.to_be_bytes().to_vec();
    p.extend_from_slice(&[0x00, 0x10]); // 16 in BE
    s.extend_from_slice(&chunk_be(b"SCDb", &p));
    s.extend_from_slice(&chunk_be(b"SCEb", &[]));
    let view = EaStreamView::parse(&s).unwrap();
    assert!(view.is_big_endian());
    assert_eq!(view.header().codec, Codec::Pcm16Be);
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
    let s = stream_ps1(Some(99), 1, &[spu_body(1, 0x29)]);
    assert_eq!(EaStreamView::parse(&s), Err(Error::UnsupportedCodec(99)));
    // Invalid channels
    let mut s2 = chunk(b"SCHl", &pt(9, Some(0x00), 22050, 0, false, &[]));
    s2.extend_from_slice(&chunk(b"SCEl", &[]));
    assert_eq!(EaStreamView::parse(&s2), Err(Error::InvalidChannels(9)));
}

#[test]
fn reset_redecodes_identically() {
    let s = stream_ps1(None, 1, &[spu_body(2, 0x29)]);
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
