//! Streaming zero-allocation decoder over an [`EaStreamView`].

use crate::codec::{
    decode_ea_xa_subframe_strided, decode_ea_xa_v1_stereo_block, decode_spu_block,
    decode_spu_block_strided, AdpcmState, Codec,
};
use crate::{ChunkKind, EaStreamView, Error, Result};

const GROUP_FRAMES: usize = 28;
const MAX_CHANNELS: usize = 2;
const INV_32768: f32 = 1.0 / 32768.0;

#[derive(Debug, Clone, Copy)]
struct LoopSnapshot<'a> {
    next_chunk_offset: usize,
    payload: &'a [u8],
    payload_pos: usize,
    payload_big_endian: bool,
    in_payload_header: bool,
    group: [i16; GROUP_FRAMES * MAX_CHANNELS],
    group_frames: usize,
    group_pos: usize,
    state: [AdpcmState; MAX_CHANNELS],
    frames_decoded: u64,
}

/// Streaming decoder. State is fixed-size; no heap allocation occurs.
#[derive(Debug, Clone)]
pub struct EaDecoder<'a> {
    view: EaStreamView<'a>,
    // Position of next chunk to load (byte offset), 0 = not started.
    next_chunk_offset: usize,
    started: bool,
    finished: bool,
    // Current SCDl/SCDb payload cursor.
    payload: &'a [u8],
    payload_pos: usize,
    payload_big_endian: bool,
    in_payload_header: bool,
    // Decoded group buffer (interleaved).
    group: [i16; GROUP_FRAMES * MAX_CHANNELS],
    group_frames: usize,
    group_pos: usize,
    state: [AdpcmState; MAX_CHANNELS],
    /// Total frames emitted since start/reset (reset to loop_start on wrap).
    frames_decoded: u64,
    loop_count: u32,
    loop_snapshot: Option<LoopSnapshot<'a>>,
}

impl<'a> EaDecoder<'a> {
    /// Create a decoder for a parsed stream view.
    #[must_use]
    pub fn new(view: EaStreamView<'a>) -> Self {
        let be = view.is_big_endian();
        Self {
            view,
            next_chunk_offset: 0,
            started: false,
            finished: false,
            payload: &[],
            payload_pos: 0,
            payload_big_endian: be,
            in_payload_header: true,
            group: [0; GROUP_FRAMES * MAX_CHANNELS],
            group_frames: 0,
            group_pos: 0,
            state: [AdpcmState::default(); MAX_CHANNELS],
            frames_decoded: 0,
            loop_count: 0,
            loop_snapshot: None,
        }
    }

    /// Channel count (1 or 2).
    #[must_use]
    pub fn channels(&self) -> u32 {
        self.view.header().channels
    }

    /// Sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.view.header().sample_rate
    }

    /// Resolved codec.
    #[must_use]
    pub fn codec(&self) -> Codec {
        self.view.header().codec
    }

    /// Frames decoded so far.
    #[must_use]
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Number of times the loop has wrapped.
    #[must_use]
    pub fn loop_count(&self) -> u32 {
        self.loop_count
    }

    /// Whether the stream is exhausted (and no loop applies).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Reset to the first data chunk, clearing predictor state and cached loop snapshot.
    pub fn reset(&mut self) {
        self.next_chunk_offset = 0;
        self.started = false;
        self.finished = false;
        self.payload = &[];
        self.payload_pos = 0;
        self.payload_big_endian = self.view.is_big_endian();
        self.in_payload_header = true;
        self.group_frames = 0;
        self.group_pos = 0;
        self.state = [AdpcmState::default(); MAX_CHANNELS];
        self.frames_decoded = 0;
        self.loop_count = 0;
        self.loop_snapshot = None;
    }

    fn loop_bounds(&self) -> Option<(u64, u64)> {
        let h = self.view.header();
        match (h.loop_start, h.loop_end) {
            (Some(s), Some(e)) if e > s => Some((u64::from(s), u64::from(e))),
            _ => None,
        }
    }

    fn capture_snapshot(&self) -> LoopSnapshot<'a> {
        LoopSnapshot {
            next_chunk_offset: self.next_chunk_offset,
            payload: self.payload,
            payload_pos: self.payload_pos,
            payload_big_endian: self.payload_big_endian,
            in_payload_header: self.in_payload_header,
            group: self.group,
            group_frames: self.group_frames,
            group_pos: self.group_pos,
            state: self.state,
            frames_decoded: self.frames_decoded,
        }
    }

    fn restore_snapshot(&mut self, snap: LoopSnapshot<'a>) {
        self.next_chunk_offset = snap.next_chunk_offset;
        self.started = true;
        self.finished = false;
        self.payload = snap.payload;
        self.payload_pos = snap.payload_pos;
        self.payload_big_endian = snap.payload_big_endian;
        self.in_payload_header = snap.in_payload_header;
        self.group = snap.group;
        self.group_frames = snap.group_frames;
        self.group_pos = snap.group_pos;
        self.state = snap.state;
        self.frames_decoded = snap.frames_decoded;
    }

    /// Advance to the next data payload; returns `false` at end of stream.
    fn load_next_payload(&mut self) -> Result<bool> {
        loop {
            let bytes = self.view.bytes();
            if !self.started {
                self.started = true;
                self.next_chunk_offset = 0;
            }
            if self.next_chunk_offset >= bytes.len() {
                return Ok(false);
            }
            let chunk = match parse_one(bytes, self.next_chunk_offset)? {
                Some(c) => c,
                None => return Ok(false),
            };
            self.next_chunk_offset = chunk.offset + chunk.size;
            match chunk.kind {
                ChunkKind::Data => {
                    self.payload =
                        &bytes[chunk.payload_offset..chunk.payload_offset + chunk.payload_len];
                    self.payload_pos = 0;
                    self.payload_big_endian = chunk.big_endian;
                    self.in_payload_header = true;
                    return Ok(true);
                }
                ChunkKind::End => return Ok(false),
                _ => continue,
            }
        }
    }

    /// Parse the codec/generation-aware `SCDl`/`SCDb` subheader at the start of `self.payload`.
    fn consume_payload_subheader(&mut self) {
        let header = self.view.header();
        let channels = header.channels as usize;
        let codec = header.codec;
        let be = self.payload_big_endian;
        let has_ps1_word = header.ps1_stream_word.is_some();

        let needed = header.scdl_subheader_len();
        if self.payload.len() < needed {
            self.payload_pos = self.payload.len();
            self.in_payload_header = false;
            return;
        }

        if matches!(
            codec,
            Codec::EaAdpcmR1 | Codec::EaAdpcmR2 | Codec::EaAdpcmR3
        ) {
            // Skip 4-byte sample count (plus optional 4-byte PS1 stream word),
            // then load per-channel (prev1, prev2) i16 initial history.
            let hist_start = if has_ps1_word { 8 } else { 4 };
            for ch in 0..channels {
                let off = hist_start + ch * 4;
                let (prev1, prev2) = if be {
                    (
                        i16::from_be_bytes([self.payload[off], self.payload[off + 1]]),
                        i16::from_be_bytes([self.payload[off + 2], self.payload[off + 3]]),
                    )
                } else {
                    (
                        i16::from_le_bytes([self.payload[off], self.payload[off + 1]]),
                        i16::from_le_bytes([self.payload[off + 2], self.payload[off + 3]]),
                    )
                };
                self.state[ch].prev1 = i32::from(prev1);
                self.state[ch].prev2 = i32::from(prev2);
            }
        }

        self.payload_pos = needed;
        self.in_payload_header = false;
    }

    /// Refill the decoded group buffer. Returns `false` at EOF.
    fn refill_group(&mut self) -> Result<bool> {
        let channels = self.channels() as usize;
        let codec = self.codec();
        loop {
            if self.payload_pos >= self.payload.len() {
                if !self.load_next_payload()? {
                    return Ok(false);
                }
                continue;
            }
            if self.in_payload_header {
                self.consume_payload_subheader();
                if self.payload_pos >= self.payload.len() {
                    continue;
                }
            }
            let avail = &self.payload[self.payload_pos..];
            match codec {
                Codec::Pcm8 | Codec::Pcm16Le | Codec::Pcm16Be => {
                    let frame_bytes = codec.pcm_frame_bytes(channels).unwrap_or(channels);
                    let frames = (avail.len() / frame_bytes).min(GROUP_FRAMES);
                    if frames == 0 {
                        // Partial trailing frame: skip it and move on.
                        self.payload_pos = self.payload.len();
                        continue;
                    }
                    let sample_bytes = frame_bytes / channels;
                    for f in 0..frames {
                        for ch in 0..channels {
                            let off = f * frame_bytes + ch * sample_bytes;
                            let v = match codec {
                                Codec::Pcm8 => i16::from(avail[off] as i8) << 8,
                                Codec::Pcm16Le => i16::from_le_bytes([avail[off], avail[off + 1]]),
                                _ => i16::from_be_bytes([avail[off], avail[off + 1]]),
                            };
                            self.group[f * channels + ch] = v;
                        }
                    }
                    self.payload_pos += frames * frame_bytes;
                    self.group_frames = frames;
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::Ps1SpuAdpcm => {
                    if channels == 1 {
                        if avail.len() < 16 {
                            self.payload_pos = self.payload.len();
                            continue;
                        }
                        let n = decode_spu_block(
                            &mut self.state[0],
                            &avail[..16],
                            &mut self.group[..28],
                        )?;
                        self.group_frames = n;
                        self.payload_pos += 16;
                    } else {
                        // Stereo: blocks alternate L, R. Decode directly with stride 2.
                        if avail.len() < 32 {
                            self.payload_pos = self.payload.len();
                            continue;
                        }
                        decode_spu_block_strided::<2>(
                            &mut self.state[0],
                            &avail[..16],
                            &mut self.group[..56],
                            0,
                        )?;
                        decode_spu_block_strided::<2>(
                            &mut self.state[1],
                            &avail[16..32],
                            &mut self.group[..56],
                            1,
                        )?;
                        self.group_frames = 28;
                        self.payload_pos += 32;
                    }
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::EaXaV1 => {
                    if channels == 1 {
                        if avail.len() < 15 {
                            self.payload_pos = self.payload.len();
                            continue;
                        }
                        decode_ea_xa_subframe_strided::<1>(
                            &mut self.state[0],
                            &avail[..15],
                            1,
                            &mut self.group[..28],
                            0,
                            true,
                        )?;
                        self.payload_pos += 15;
                    } else {
                        // EA-XA v1 stereo: 30-byte block (2 header bytes + 28 L/R nibble-interleaved bytes).
                        if avail.len() < 30 {
                            self.payload_pos = self.payload.len();
                            continue;
                        }
                        let (sl, sr) = self.state.split_at_mut(1);
                        decode_ea_xa_v1_stereo_block(
                            &mut sl[0],
                            &mut sr[0],
                            &avail[..30],
                            &mut self.group[..56],
                        )?;
                        self.payload_pos += 30;
                    }
                    self.group_frames = 28;
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::EaXaV2 | Codec::EaAdpcmR1 | Codec::EaAdpcmR2 | Codec::EaAdpcmR3 => {
                    let high_first = matches!(codec, Codec::EaAdpcmR2);
                    let subframe_len = if matches!(codec, Codec::EaAdpcmR3) {
                        16
                    } else {
                        15
                    };
                    let needed = subframe_len * channels;
                    if avail.len() < needed {
                        self.payload_pos = self.payload.len();
                        continue;
                    }
                    let data_offset = if matches!(codec, Codec::EaAdpcmR3) {
                        2
                    } else {
                        1
                    };
                    if channels == 1 {
                        decode_ea_xa_subframe_strided::<1>(
                            &mut self.state[0],
                            &avail[..subframe_len],
                            data_offset,
                            &mut self.group[..28],
                            0,
                            high_first,
                        )?;
                    } else {
                        decode_ea_xa_subframe_strided::<2>(
                            &mut self.state[0],
                            &avail[..subframe_len],
                            data_offset,
                            &mut self.group[..56],
                            0,
                            high_first,
                        )?;
                        decode_ea_xa_subframe_strided::<2>(
                            &mut self.state[1],
                            &avail[subframe_len..subframe_len * 2],
                            data_offset,
                            &mut self.group[..56],
                            1,
                            high_first,
                        )?;
                    }
                    self.group_frames = 28;
                    self.payload_pos += needed;
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::Mt10 => return Err(Error::UnsupportedCodec(0x04)),
                Codec::EaXas => return Err(Error::UnsupportedCodec(0x16)),
                Codec::Unknown(id) => return Err(Error::UnsupportedCodec(id)),
            }
        }
    }

    /// Ensure `self.group` has frames ready to emit, handling loop-start snapshots and loop-end wraps.
    /// Returns the number of frames to consume from `self.group[self.group_pos..]` (`0` at EOF).
    fn prepare_frames(&mut self, max_take: usize) -> Result<usize> {
        loop {
            if let Some((s, e)) = self.loop_bounds() {
                if self.frames_decoded >= e {
                    self.seek_to_loop_start(s)?;
                    self.loop_count = self.loop_count.saturating_add(1);
                }
            }
            if self.group_pos >= self.group_frames {
                if self.finished {
                    return Ok(0);
                }
                if !self.refill_group()? {
                    if let Some((s, _e)) = self.loop_bounds() {
                        if self.frames_decoded > s {
                            self.seek_to_loop_start(s)?;
                            self.loop_count = self.loop_count.saturating_add(1);
                            continue;
                        }
                    }
                    self.finished = true;
                    return Ok(0);
                }
            }
            let avail_frames = self.group_frames - self.group_pos;
            let mut take = avail_frames.min(max_take);
            if let Some((s, e)) = self.loop_bounds() {
                if self.loop_snapshot.is_none() {
                    if self.frames_decoded == s {
                        self.loop_snapshot = Some(self.capture_snapshot());
                    } else if self.frames_decoded < s {
                        let to_start = (s - self.frames_decoded) as usize;
                        take = take.min(to_start);
                    }
                }
                let remaining = e.saturating_sub(self.frames_decoded) as usize;
                take = take.min(remaining);
                if take == 0 {
                    continue;
                }
            }
            return Ok(take);
        }
    }

    /// Decode interleaved `i16` samples into `out`.
    /// Returns the number of sample frames (per channel) written.
    pub fn decode_i16(&mut self, out: &mut [i16]) -> Result<usize> {
        let channels = self.channels() as usize;
        if channels == 0 || out.len() < channels {
            return Ok(0);
        }
        let max_frames = out.len() / channels;
        let mut written = 0usize;
        while written < max_frames {
            let take = self.prepare_frames(max_frames - written)?;
            if take == 0 {
                break;
            }
            let src = self.group_pos * channels;
            let dst = written * channels;
            out[dst..dst + take * channels]
                .copy_from_slice(&self.group[src..src + take * channels]);
            self.group_pos += take;
            written += take;
            self.frames_decoded += take as u64;
        }
        Ok(written)
    }

    /// Decode into separate planar `f32` left/right buffers directly from the group buffer.
    /// For mono, the signal is written to `left` and `right` is filled
    /// with the same samples. Returns frames written.
    pub fn decode_f32_planar(&mut self, left: &mut [f32], right: &mut [f32]) -> Result<usize> {
        let n = left.len().min(right.len());
        let channels = self.channels() as usize;
        if channels == 0 || n == 0 {
            return Ok(0);
        }
        let mut done = 0usize;
        while done < n {
            let take = self.prepare_frames(n - done)?;
            if take == 0 {
                break;
            }
            if channels == 1 {
                let src = &self.group[self.group_pos..self.group_pos + take];
                for (i, &s) in src.iter().enumerate() {
                    let v = f32::from(s) * INV_32768;
                    left[done + i] = v;
                    right[done + i] = v;
                }
            } else {
                let src = &self.group[self.group_pos * 2..(self.group_pos + take) * 2];
                for i in 0..take {
                    left[done + i] = f32::from(src[i * 2]) * INV_32768;
                    right[done + i] = f32::from(src[i * 2 + 1]) * INV_32768;
                }
            }
            self.group_pos += take;
            done += take;
            self.frames_decoded += take as u64;
        }
        Ok(done)
    }

    /// Jump back to `loop_start` in O(1) using the cached snapshot, falling back to a linear
    /// scan only if the snapshot was not yet captured.
    fn seek_to_loop_start(&mut self, target: u64) -> Result<()> {
        if let Some(snap) = self.loop_snapshot {
            self.restore_snapshot(snap);
            return Ok(());
        }
        let saved_loop = self.loop_count;
        self.reset();
        self.loop_count = saved_loop;
        let mut remaining = target;
        while remaining > 0 {
            if self.group_pos >= self.group_frames && !self.refill_group()? {
                break;
            }
            let avail = (self.group_frames - self.group_pos) as u64;
            let take = avail.min(remaining) as usize;
            self.group_pos += take;
            self.frames_decoded += take as u64;
            remaining -= take as u64;
        }
        self.loop_snapshot = Some(self.capture_snapshot());
        Ok(())
    }
}

fn parse_one(bytes: &[u8], offset: usize) -> Result<Option<crate::Chunk>> {
    if offset >= bytes.len() {
        return Ok(None);
    }
    let remaining = bytes.len() - offset;
    if remaining < 8 {
        return Err(Error::Truncated {
            offset,
            needed: 8,
            available: remaining,
        });
    }
    let tag = [
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ];
    let be = tag[3] == b'b';
    let raw = if be {
        u32::from_be_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ])
    } else {
        u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ])
    };
    if raw < 8 {
        return Err(Error::InvalidChunkSize { offset, size: raw });
    }
    let size = raw as usize;
    if offset + size > bytes.len() {
        return Err(Error::Truncated {
            offset,
            needed: size,
            available: remaining,
        });
    }
    let kind = match &tag {
        b"SCHl" | b"SCHb" => ChunkKind::Header,
        b"SCCl" | b"SCCb" => ChunkKind::Count,
        b"SCDl" | b"SCDb" => ChunkKind::Data,
        b"SCLl" | b"SCLb" => ChunkKind::Loop,
        b"SCEl" | b"SCEb" => ChunkKind::End,
        _ => ChunkKind::Other(tag),
    };
    Ok(Some(crate::Chunk {
        kind,
        tag,
        offset,
        size,
        payload_offset: offset + 8,
        payload_len: size - 8,
        big_endian: be,
    }))
}
