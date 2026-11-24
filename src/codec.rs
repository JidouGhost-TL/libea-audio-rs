//! Core PCM / ADPCM sample decoders. Hot loops perform no heap allocation.

use crate::{Error, Result};

/// EA-XA predictor table `(coef1, coef2)`, denominator 256.
pub const EA_XA_PREDICTORS: [(i32, i32); 4] = [(0, 0), (240, 0), (460, -208), (392, -220)];
/// PS1 SPU-ADPCM predictor table `(coef1, coef2)`, denominator 64.
pub const SPU_PREDICTORS: [(i32, i32); 5] = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)];

/// Codec identifier carried in PT TLV tag `0x83` (or inferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Codec {
    /// Inferred from a PS1 stream word: SPU-ADPCM blocks inside `SCDl`.
    Ps1SpuAdpcm,
    /// EA-XA version 1, 4-bit ADPCM.
    EaXaV1,
    /// EA-XA version 2, 4-bit ADPCM.
    #[default]
    EaXaV2,
    /// Signed 8-bit PCM.
    Pcm8,
    /// Signed 16-bit little-endian PCM.
    Pcm16Le,
    /// Signed 16-bit big-endian PCM.
    Pcm16Be,
    /// EA-ADPCM R1 frame variant (low nibble first).
    EaAdpcmR1,
    /// EA-ADPCM R2 frame variant (high nibble first).
    EaAdpcmR2,
    /// EA-ADPCM R3 frame variant (16-byte subframe, reserved byte).
    EaAdpcmR3,
    /// Unknown raw codec id from the PT header.
    Unknown(u32),
}

impl Codec {
    /// Map a PT codec id to a known codec. `None` means infer from context.
    pub fn from_pt_id(id: u32) -> Self {
        match id {
            1 => Self::Pcm8,
            2 => Self::Pcm16Le,
            3 => Self::Pcm16Be,
            4 => Self::EaXaV1,
            5 => Self::EaXaV2,
            6 => Self::Ps1SpuAdpcm,
            7 => Self::EaAdpcmR1,
            8 => Self::EaAdpcmR2,
            9 => Self::EaAdpcmR3,
            other => Self::Unknown(other),
        }
    }
    /// Bytes per sample frame (all channels) for PCM codecs.
    pub fn pcm_frame_bytes(self, channels: usize) -> Option<usize> {
        match self {
            Self::Pcm8 => Some(channels),
            Self::Pcm16Le | Self::Pcm16Be => Some(channels * 2),
            _ => None,
        }
    }
    /// Whether this codec uses ADPCM group framing.
    pub fn is_adpcm(self) -> bool {
        !matches!(self, Self::Pcm8 | Self::Pcm16Le | Self::Pcm16Be)
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

#[inline]
fn clamp_i16(v: i32) -> i16 {
    v.clamp(-32768, 32767) as i16
}

#[inline]
fn signed_nibble(n: u8) -> i32 {
    let n = (n & 0x0F) as i32;
    if n >= 8 {
        n - 16
    } else {
        n
    }
}

/// Decode one EA-XA sample. `shift = 20 - (hdr & 0xF)` per the EA-XA spec;
/// the predictor index is the high nibble masked to the 4-entry table.
#[inline]
pub fn ea_xa_step(state: &mut AdpcmState, hdr: u8, nibble: u8) -> i16 {
    let (c1, c2) = EA_XA_PREDICTORS[usize::from((hdr >> 4) & 0x03)];
    let shift = 20i32 - i32::from(hdr & 0x0F);
    let base = (signed_nibble(nibble) << shift) >> 8;
    let pred = (state.prev1 * c1 + state.prev2 * c2) >> 8;
    let v = clamp_i16(base + pred);
    state.push(i32::from(v));
    v
}

/// Decode one PS1 SPU-ADPCM sample. Filter is the high nibble (0..=4),
/// shift is the low nibble; nibbles are consumed low-first by callers.
#[inline]
pub fn spu_step(state: &mut AdpcmState, hdr: u8, nibble: u8) -> i16 {
    let idx = usize::from(hdr >> 4);
    let (c1, c2) = if idx < SPU_PREDICTORS.len() {
        SPU_PREDICTORS[idx]
    } else {
        (0, 0)
    };
    let shift = i32::from(hdr & 0x0F);
    let n = signed_nibble(nibble);
    let base = if shift <= 12 {
        n << (12 - shift)
    } else {
        n >> (shift - 12)
    };
    let pred = (state.prev1 * c1 + state.prev2 * c2 + 32) >> 6;
    let v = clamp_i16(base + pred);
    state.push(i32::from(v));
    v
}

/// Decode a 16-byte PS1 SPU-ADPCM block (28 samples) into `out`.
/// Byte 0 is filter/shift, byte 1 is flags (ignored), bytes 2..16 hold
/// 28 nibbles, low nibble first. Returns samples written (28).
pub fn decode_spu_block(state: &mut AdpcmState, block: &[u8], out: &mut [i16]) -> Result<usize> {
    if block.len() < 16 {
        return Err(Error::TruncatedData {
            needed: 16,
            available: block.len(),
        });
    }
    if out.len() < 28 {
        return Err(Error::OutputTooSmall {
            needed: 28,
            available: out.len(),
        });
    }
    let hdr = block[0];
    for i in 0..14 {
        let b = block[2 + i];
        out[i * 2] = spu_step(state, hdr, b & 0x0F);
        out[i * 2 + 1] = spu_step(state, hdr, b >> 4);
    }
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
    if frame.len() < data_offset + 14 {
        return Err(Error::TruncatedData {
            needed: data_offset + 14,
            available: frame.len(),
        });
    }
    if out.len() < 28 {
        return Err(Error::OutputTooSmall {
            needed: 28,
            available: out.len(),
        });
    }
    let hdr = frame[0];
    for i in 0..14 {
        let b = frame[data_offset + i];
        let (lo, hi) = if high_first {
            (b >> 4, b & 0x0F)
        } else {
            (b & 0x0F, b >> 4)
        };
        // `lo` is the first nibble in stream order.
        out[i * 2] = ea_xa_step(state, hdr, lo);
        out[i * 2 + 1] = ea_xa_step(state, hdr, hi);
    }
    Ok(28)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ea_xa_silence_and_clamp() {
        let mut st = AdpcmState::default();
        // Header 0x00: predictor (0,0), shift 20 -> nibble 0 decodes to 0.
        assert_eq!(ea_xa_step(&mut st, 0x00, 0), 0);
        // Max positive nibble with largest shift stays in range and positive.
        let mut st = AdpcmState::default();
        let v = ea_xa_step(&mut st, 0x00, 7);
        assert!(v > 0);
    }

    #[test]
    fn spu_block_roundtrip_shape() {
        let mut block = [0u8; 16];
        block[0] = 0x29; // filter 2, shift 9
        block[1] = 0x02;
        for b in block[2..].iter_mut() {
            *b = 0x11;
        }
        let mut st = AdpcmState::default();
        let mut out = [0i16; 28];
        assert_eq!(decode_spu_block(&mut st, &block, &mut out).unwrap(), 28);
        assert!(out.iter().all(|&s| s != i16::MIN));
    }

    #[test]
    fn truncated_inputs_error() {
        let mut st = AdpcmState::default();
        let mut out = [0i16; 28];
        assert!(decode_spu_block(&mut st, &[0u8; 4], &mut out).is_err());
        assert!(decode_ea_xa_subframe(&mut st, &[0u8; 4], &mut out, false).is_err());
    }
}
