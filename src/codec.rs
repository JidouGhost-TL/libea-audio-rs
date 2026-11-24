//! Core PCM / ADPCM sample decoders. Hot loops perform no heap allocation.

use crate::{Error, Result};

/// EA-XA predictor table `(coef1, coef2)`, denominator 256.
pub const EA_XA_PREDICTORS: [(i32, i32); 4] = [(0, 0), (240, 0), (460, -208), (392, -220)];
/// PS1 SPU-ADPCM predictor table `(coef1, coef2)`, denominator 64.
pub const SPU_PREDICTORS: [(i32, i32); 5] = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)];

/// Codec identifier carried in PT TLV tag `0x83` (or inferred from stream context).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Codec {
    /// PS1 SPU-ADPCM blocks inside `SCDl` (inferred from a PS1 stream word or `0x83 = 0x05` on PS1).
    Ps1SpuAdpcm,
    /// EA-XA version 1, 4-bit ADPCM (`0x83 = 0x05` on non-PS1; stereo uses 30-byte interleaved blocks).
    EaXaV1,
    /// EA-XA version 2, 4-bit ADPCM (`0x83 = 0x07` with version 0/absent; 15-byte mono subframes).
    #[default]
    EaXaV2,
    /// Signed 8-bit PCM (`0x83 = 0x00` with 8-bit sample width in tag `0x81`).
    Pcm8,
    /// Signed 16-bit little-endian PCM (`0x83 = 0x00` in `SCHl`).
    Pcm16Le,
    /// Signed 16-bit big-endian PCM (`0x83 = 0x00` in `SCHb`).
    Pcm16Be,
    /// EA-ADPCM R1 frame variant (`0x83 = 0x07`, `0x80 = 1`; low nibble first, per-chunk history).
    EaAdpcmR1,
    /// EA-ADPCM R2 frame variant (`0x83 = 0x07`, `0x80 = 2`; high nibble first, per-chunk history).
    EaAdpcmR2,
    /// EA-ADPCM R3 frame variant (`0x83 = 0x07`, `0x80 = 3`; 16-byte subframe, per-chunk history).
    EaAdpcmR3,
    /// EA MT10 / MicroTalk (`0x83 = 0x04`, metadata-only; decoding returns [`Error::UnsupportedCodec`]).
    Mt10,
    /// EA-XAS (`0x83 = 0x16`, metadata-only; decoding returns [`Error::UnsupportedCodec`]).
    EaXas,
    /// Unknown raw codec id from PT tag `0x83`.
    Unknown(u32),
}

impl Codec {
    /// Map a raw EA PT tag `0x83` compression id to a [`Codec`] without container context.
    ///
    /// For full disambiguation (PCM width/endianness via tag `0x81` and `SCHl`/`SCHb`,
    /// PS1 SPU-ADPCM vs EA-XA v1 via PS1 stream word / tag `0x80`, and EA-ADPCM R1/R2/R3
    /// via tag `0x80`), use [`Codec::from_pt_context`].
    #[must_use]
    pub fn from_pt_id(id: u32) -> Self {
        match id {
            0x00 => Self::Pcm16Le,
            0x04 => Self::Mt10,
            0x05 => Self::EaXaV1,
            0x07 => Self::EaXaV2,
            0x16 => Self::EaXas,
            other => Self::Unknown(other),
        }
    }

    /// Resolve a [`Codec`] from full EA PT header context.
    ///
    /// - `codec_id`: PT tag `0x83` compression ID (`None` if omitted).
    /// - `version`: PT tag `0x80` version/platform field (`None` if omitted).
    /// - `bits_per_sample`: PT tag `0x81` sample width (`1` or `8` for 8-bit, `2` or `16` for 16-bit).
    /// - `ps1_stream_word`: Leading 4-byte PS1 stream word before `PT`, if present.
    /// - `big_endian`: `true` for big-endian `SCHb` streams, `false` for little-endian `SCHl`.
    #[must_use]
    pub fn from_pt_context(
        codec_id: Option<u32>,
        version: Option<u32>,
        bits_per_sample: Option<u32>,
        ps1_stream_word: Option<u32>,
        big_endian: bool,
    ) -> Self {
        match codec_id {
            Some(0x00) => match bits_per_sample {
                Some(1) | Some(8) => Self::Pcm8,
                _ if big_endian => Self::Pcm16Be,
                _ => Self::Pcm16Le,
            },
            Some(0x04) => Self::Mt10,
            Some(0x05) => {
                if ps1_stream_word.is_some() || version == Some(0x05) {
                    Self::Ps1SpuAdpcm
                } else {
                    Self::EaXaV1
                }
            }
            Some(0x07) => match version {
                Some(1) => Self::EaAdpcmR1,
                Some(2) => Self::EaAdpcmR2,
                Some(3) => Self::EaAdpcmR3,
                _ => Self::EaXaV2,
            },
            Some(0x16) => Self::EaXas,
            Some(other) => Self::Unknown(other),
            None => {
                if ps1_stream_word.is_some() {
                    Self::Ps1SpuAdpcm
                } else {
                    Self::EaXaV2
                }
            }
        }
    }

    /// Bytes per sample frame (all channels) for uncompressed PCM codecs.
    #[must_use]
    pub fn pcm_frame_bytes(self, channels: usize) -> Option<usize> {
        match self {
            Self::Pcm8 => Some(channels),
            Self::Pcm16Le | Self::Pcm16Be => Some(channels * 2),
            _ => None,
        }
    }

    /// Whether this codec uses ADPCM group framing.
    #[must_use]
    pub fn is_adpcm(self) -> bool {
        matches!(
            self,
            Self::Ps1SpuAdpcm
                | Self::EaXaV1
                | Self::EaXaV2
                | Self::EaAdpcmR1
                | Self::EaAdpcmR2
                | Self::EaAdpcmR3
                | Self::EaXas
        )
    }
}

/// Per-channel ADPCM predictor state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdpcmState {
    /// Most recent decoded sample.
    pub prev1: i32,
    /// Second most recent decoded sample.
    pub prev2: i32,
}

impl AdpcmState {
    /// Reset predictor history to silence.
    pub fn reset(&mut self) {
        self.prev1 = 0;
        self.prev2 = 0;
    }

    #[inline]
    fn push(&mut self, s: i32) {
        self.prev2 = self.prev1;
        self.prev1 = s;
    }
}

#[inline(always)]
fn clamp_i16(v: i32) -> i16 {
    v.clamp(-32768, 32767) as i16
}

#[inline(always)]
fn signed_nibble(n: u8) -> i32 {
    let n = (n & 0x0F) as i32;
    if n >= 8 {
        n - 16
    } else {
        n
    }
}

#[inline(always)]
fn ea_xa_params(hdr: u8) -> (i32, i32, i32) {
    let (c1, c2) = EA_XA_PREDICTORS[usize::from((hdr >> 4) & 0x03)];
    let shift = 20i32 - i32::from(hdr & 0x0F);
    (c1, c2, shift)
}

#[inline(always)]
fn ea_xa_step_inner(prev1: i32, prev2: i32, c1: i32, c2: i32, shift: i32, nibble: u8) -> i16 {
    let base = (signed_nibble(nibble) << shift) >> 8;
    let pred = (prev1 * c1 + prev2 * c2 + 128) >> 8;
    clamp_i16(base + pred)
}

#[inline(always)]
fn spu_params(hdr: u8) -> (i32, i32, i32) {
    let idx = usize::from(hdr >> 4);
    let (c1, c2) = if idx < SPU_PREDICTORS.len() {
        SPU_PREDICTORS[idx]
    } else {
        (0, 0)
    };
    let shift = i32::from(hdr & 0x0F);
    (c1, c2, shift)
}

#[inline(always)]
fn spu_step_inner(prev1: i32, prev2: i32, c1: i32, c2: i32, shift: i32, nibble: u8) -> i16 {
    let n = signed_nibble(nibble);
    let base = if shift <= 12 {
        n << (12 - shift)
    } else {
        n >> (shift - 12)
    };
    let pred = (prev1 * c1 + prev2 * c2 + 32) >> 6;
    clamp_i16(base + pred)
}

/// Decode one EA-XA sample. `shift = 20 - (hdr & 0xF)` per the EA-XA spec;
/// the predictor index is the high nibble masked to the 4-entry table, with
/// hardware `+ 128` rounding in the predictor term.
#[inline]
pub fn ea_xa_step(state: &mut AdpcmState, hdr: u8, nibble: u8) -> i16 {
    let (c1, c2, shift) = ea_xa_params(hdr);
    let v = ea_xa_step_inner(state.prev1, state.prev2, c1, c2, shift, nibble);
    state.push(i32::from(v));
    v
}

/// Decode one PS1 SPU-ADPCM sample. Filter is the high nibble (0..=4),
/// shift is the low nibble; nibbles are consumed low-first by callers.
#[inline]
pub fn spu_step(state: &mut AdpcmState, hdr: u8, nibble: u8) -> i16 {
    let (c1, c2, shift) = spu_params(hdr);
    let v = spu_step_inner(state.prev1, state.prev2, c1, c2, shift, nibble);
    state.push(i32::from(v));
    v
}

/// Decode a 16-byte PS1 SPU-ADPCM block (28 samples) into `out`.
/// Byte 0 is filter/shift, byte 1 is flags (ignored), bytes 2..16 hold
/// 28 nibbles, low nibble first. Returns samples written (28).
pub fn decode_spu_block(state: &mut AdpcmState, block: &[u8], out: &mut [i16]) -> Result<usize> {
    decode_spu_block_strided::<1>(state, block, out, 0)
}

/// Decode a 16-byte PS1 SPU-ADPCM block directly into `out` starting at `offset` with `STRIDE`.
#[inline]
pub(crate) fn decode_spu_block_strided<const STRIDE: usize>(
    state: &mut AdpcmState,
    block: &[u8],
    out: &mut [i16],
    offset: usize,
) -> Result<usize> {
    if block.len() < 16 {
        return Err(Error::TruncatedData {
            needed: 16,
            available: block.len(),
        });
    }
    let needed = offset + 27 * STRIDE + 1;
    if out.len() < needed {
        return Err(Error::OutputTooSmall {
            needed,
            available: out.len(),
        });
    }
    let (c1, c2, shift) = spu_params(block[0]);
    let mut prev1 = state.prev1;
    let mut prev2 = state.prev2;
    let out_slice = &mut out[offset..needed];
    let nibbles = &block[2..16];
    if shift <= 12 {
        let lshift = 12 - shift;
        for (i, &b) in nibbles.iter().enumerate() {
            let base0 = signed_nibble(b & 0x0F) << lshift;
            let s0 = clamp_i16(base0 + ((prev1 * c1 + prev2 * c2 + 32) >> 6));
            prev2 = prev1;
            prev1 = i32::from(s0);
            out_slice[(i * 2) * STRIDE] = s0;

            let base1 = signed_nibble(b >> 4) << lshift;
            let s1 = clamp_i16(base1 + ((prev1 * c1 + prev2 * c2 + 32) >> 6));
            prev2 = prev1;
            prev1 = i32::from(s1);
            out_slice[(i * 2 + 1) * STRIDE] = s1;
        }
    } else {
        let rshift = shift - 12;
        for (i, &b) in nibbles.iter().enumerate() {
            let base0 = signed_nibble(b & 0x0F) >> rshift;
            let s0 = clamp_i16(base0 + ((prev1 * c1 + prev2 * c2 + 32) >> 6));
            prev2 = prev1;
            prev1 = i32::from(s0);
            out_slice[(i * 2) * STRIDE] = s0;

            let base1 = signed_nibble(b >> 4) >> rshift;
            let s1 = clamp_i16(base1 + ((prev1 * c1 + prev2 * c2 + 32) >> 6));
            prev2 = prev1;
            prev1 = i32::from(s1);
            out_slice[(i * 2 + 1) * STRIDE] = s1;
        }
    }
    state.prev1 = prev1;
    state.prev2 = prev2;
    Ok(28)
}

/// Decode a 15-byte EA-XA subframe (28 mono samples) into `out`.
/// Byte 0 is the predictor/shift header, bytes 1..15 hold 28 nibbles,
/// low nibble first (v2 / R1 order) or high nibble first when `high_first`.
pub fn decode_ea_xa_subframe(
    state: &mut AdpcmState,
    frame: &[u8],
    out: &mut [i16],
    high_first: bool,
) -> Result<usize> {
    decode_ea_xa_subframe_at(state, frame, 1, out, high_first)
}

/// Like [`decode_ea_xa_subframe`] but with an explicit data offset
/// (2 for EA-ADPCM R3 subframes: header, reserved, then data).
pub fn decode_ea_xa_subframe_at(
    state: &mut AdpcmState,
    frame: &[u8],
    data_offset: usize,
    out: &mut [i16],
    high_first: bool,
) -> Result<usize> {
    decode_ea_xa_subframe_strided::<1>(state, frame, data_offset, out, 0, high_first)
}

/// Decode an EA-XA / EA-ADPCM mono subframe directly into `out` at `offset` with `STRIDE`.
#[inline]
pub(crate) fn decode_ea_xa_subframe_strided<const STRIDE: usize>(
    state: &mut AdpcmState,
    frame: &[u8],
    data_offset: usize,
    out: &mut [i16],
    offset: usize,
    high_first: bool,
) -> Result<usize> {
    if frame.len() < data_offset + 14 {
        return Err(Error::TruncatedData {
            needed: data_offset + 14,
            available: frame.len(),
        });
    }
    let needed = offset + 27 * STRIDE + 1;
    if out.len() < needed {
        return Err(Error::OutputTooSmall {
            needed,
            available: out.len(),
        });
    }
    let (c1, c2, shift) = ea_xa_params(frame[0]);
    let mut prev1 = state.prev1;
    let mut prev2 = state.prev2;
    let out_slice = &mut out[offset..needed];
    let data = &frame[data_offset..data_offset + 14];
    if high_first {
        for (i, &b) in data.iter().enumerate() {
            let s0 = ea_xa_step_inner(prev1, prev2, c1, c2, shift, b >> 4);
            prev2 = prev1;
            prev1 = i32::from(s0);
            out_slice[(i * 2) * STRIDE] = s0;

            let s1 = ea_xa_step_inner(prev1, prev2, c1, c2, shift, b & 0x0F);
            prev2 = prev1;
            prev1 = i32::from(s1);
            out_slice[(i * 2 + 1) * STRIDE] = s1;
        }
    } else {
        for (i, &b) in data.iter().enumerate() {
            let s0 = ea_xa_step_inner(prev1, prev2, c1, c2, shift, b & 0x0F);
            prev2 = prev1;
            prev1 = i32::from(s0);
            out_slice[(i * 2) * STRIDE] = s0;

            let s1 = ea_xa_step_inner(prev1, prev2, c1, c2, shift, b >> 4);
            prev2 = prev1;
            prev1 = i32::from(s1);
            out_slice[(i * 2 + 1) * STRIDE] = s1;
        }
    }
    state.prev1 = prev1;
    state.prev2 = prev2;
    Ok(28)
}

/// Decode a 30-byte EA-XA v1 stereo block (28 interleaved stereo frames = 56 `i16` samples).
///
/// Layout:
/// - Byte `0`: left channel predictor/shift header.
/// - Byte `1`: right channel predictor/shift header.
/// - Bytes `2..30`: 28 nibble-interleaved sample bytes (high nibble = left, low nibble = right).
///
/// Returns the number of stereo frames written (`28`, occupying `56` elements of `out`).
pub fn decode_ea_xa_v1_stereo_block(
    state_l: &mut AdpcmState,
    state_r: &mut AdpcmState,
    block: &[u8],
    out: &mut [i16],
) -> Result<usize> {
    if block.len() < 30 {
        return Err(Error::TruncatedData {
            needed: 30,
            available: block.len(),
        });
    }
    if out.len() < 56 {
        return Err(Error::OutputTooSmall {
            needed: 56,
            available: out.len(),
        });
    }
    let (c1_l, c2_l, shift_l) = ea_xa_params(block[0]);
    let (c1_r, c2_r, shift_r) = ea_xa_params(block[1]);
    let mut l1 = state_l.prev1;
    let mut l2 = state_l.prev2;
    let mut r1 = state_r.prev1;
    let mut r2 = state_r.prev2;

    for (i, &b) in block[2..30].iter().enumerate() {
        let sl = ea_xa_step_inner(l1, l2, c1_l, c2_l, shift_l, b >> 4);
        l2 = l1;
        l1 = i32::from(sl);
        out[i * 2] = sl;

        let sr = ea_xa_step_inner(r1, r2, c1_r, c2_r, shift_r, b & 0x0F);
        r2 = r1;
        r1 = i32::from(sr);
        out[i * 2 + 1] = sr;
    }

    state_l.prev1 = l1;
    state_l.prev2 = l2;
    state_r.prev1 = r1;
    state_r.prev2 = r2;
    Ok(28)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ea_xa_silence_and_rounding() {
        let mut st = AdpcmState::default();
        // Header 0x00: predictor (0,0), shift 20 -> nibble 0 decodes to 0.
        assert_eq!(ea_xa_step(&mut st, 0x00, 0), 0);
        // Max positive nibble with largest shift stays in range and positive.
        let mut st = AdpcmState::default();
        let v = ea_xa_step(&mut st, 0x00, 7);
        assert!(v > 0);

        // Verify hardware +128 rounding: prev1 = 1, c1 = 240 -> (240 + 128) >> 8 = 1 (not 0).
        let mut st = AdpcmState { prev1: 1, prev2: 0 };
        assert_eq!(ea_xa_step(&mut st, 0x10, 0), 1);
    }

    #[test]
    fn ea_xa_v1_stereo_block_decodes_interleaved_channels() {
        let mut block = [0u8; 30];
        block[0] = 0x18; // L: pred 1, shift 12
        block[1] = 0x29; // R: pred 2, shift 11
        for b in &mut block[2..30] {
            *b = 0x2F; // L nibble = +2, R nibble = -1
        }
        let mut sl = AdpcmState::default();
        let mut sr = AdpcmState::default();
        let mut out = [0i16; 56];
        assert_eq!(
            decode_ea_xa_v1_stereo_block(&mut sl, &mut sr, &block, &mut out).unwrap(),
            28
        );
        // L samples positive, R samples negative on first frame.
        assert!(out[0] > 0);
        assert!(out[1] < 0);

        // Must match step-by-step ea_xa_step calls exactly.
        let mut ref_l = AdpcmState::default();
        let mut ref_r = AdpcmState::default();
        for i in 0..28 {
            assert_eq!(out[i * 2], ea_xa_step(&mut ref_l, block[0], 0x02));
            assert_eq!(out[i * 2 + 1], ea_xa_step(&mut ref_r, block[1], 0x0F));
        }
    }

    #[test]
    fn spu_block_matches_step_by_step() {
        let mut block = [0u8; 16];
        block[0] = 0x29; // filter 2, shift 9
        block[1] = 0x02;
        for (i, b) in block[2..].iter_mut().enumerate() {
            *b = ((i as u8) << 4) | ((i as u8 + 3) & 0x0F);
        }
        let mut st = AdpcmState::default();
        let mut out = [0i16; 28];
        assert_eq!(decode_spu_block(&mut st, &block, &mut out).unwrap(), 28);

        let mut ref_st = AdpcmState::default();
        for i in 0..14 {
            let b = block[2 + i];
            assert_eq!(out[i * 2], spu_step(&mut ref_st, block[0], b & 0x0F));
            assert_eq!(out[i * 2 + 1], spu_step(&mut ref_st, block[0], b >> 4));
        }
        assert_eq!(st, ref_st);
    }

    #[test]
    fn codec_pt_context_mapping() {
        assert_eq!(
            Codec::from_pt_context(Some(0x00), None, Some(1), None, false),
            Codec::Pcm8
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x00), None, Some(8), None, false),
            Codec::Pcm8
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x00), None, Some(2), None, false),
            Codec::Pcm16Le
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x00), None, Some(16), None, true),
            Codec::Pcm16Be
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x04), None, None, None, false),
            Codec::Mt10
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x05), None, None, None, false),
            Codec::EaXaV1
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x05), None, None, Some(0x1234), false),
            Codec::Ps1SpuAdpcm
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x05), Some(0x05), None, None, false),
            Codec::Ps1SpuAdpcm
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x07), None, None, None, false),
            Codec::EaXaV2
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x07), Some(1), None, None, false),
            Codec::EaAdpcmR1
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x07), Some(2), None, None, false),
            Codec::EaAdpcmR2
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x07), Some(3), None, None, false),
            Codec::EaAdpcmR3
        );
        assert_eq!(
            Codec::from_pt_context(Some(0x16), None, None, None, false),
            Codec::EaXas
        );
        assert_eq!(
            Codec::from_pt_context(None, None, None, Some(0x1234), false),
            Codec::Ps1SpuAdpcm
        );
    }

    #[test]
    fn truncated_inputs_error() {
        let mut st = AdpcmState::default();
        let mut st2 = AdpcmState::default();
        let mut out = [0i16; 56];
        assert!(decode_spu_block(&mut st, &[0u8; 4], &mut out).is_err());
        assert!(decode_ea_xa_subframe(&mut st, &[0u8; 4], &mut out, false).is_err());
        assert!(decode_ea_xa_v1_stereo_block(&mut st, &mut st2, &[0u8; 29], &mut out).is_err());
    }
}
