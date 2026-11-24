//! Streaming zero-allocation decoder over an [`EaStreamView`].

use crate::codec::{decode_ea_xa_subframe_at, decode_spu_block, AdpcmState, Codec};
use crate::{ChunkKind, EaStreamView, Error, Result, SCDL_HEADER_LEN};

const GROUP_FRAMES: usize = 28;
const MAX_CHANNELS: usize = 2;

/// Streaming decoder. State is fixed-size; no heap allocation occurs.
#[derive(Debug, Clone)]
pub struct EaDecoder<'a> {
    view: EaStreamView<'a>,
    // Position of next chunk to load (byte offset), 0 = not started.
    next_chunk_offset: usize,
    started: bool,
    finished: bool,
    // Current SCDl payload cursor.
    payload: &'a [u8],
    payload_pos: usize,
    in_payload_header: bool,
    // Decoded group buffer (interleaved).
    group: [i16; GROUP_FRAMES * MAX_CHANNELS],
    group_frames: usize,
    group_pos: usize,
    state: [AdpcmState; MAX_CHANNELS],
    /// Total frames emitted since start/reset (excluding loop repeats wrap).
    frames_decoded: u64,
    loop_count: u32,
}

impl<'a> EaDecoder<'a> {
    /// Create a decoder for a parsed stream view.
    pub fn new(view: EaStreamView<'a>) -> Self {
        Self {
            view,
            next_chunk_offset: 0,
            started: false,
            finished: false,
            payload: &[],
            payload_pos: 0,
            in_payload_header: true,
            group: [0; GROUP_FRAMES * MAX_CHANNELS],
            group_frames: 0,
            group_pos: 0,
            state: [AdpcmState::default(); MAX_CHANNELS],
            frames_decoded: 0,
            loop_count: 0,
        }
    }

    /// Channel count (1 or 2).
    pub fn channels(&self) -> u32 {
        self.view.header().channels
    }

    /// Sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.view.header().sample_rate
    }

    /// Resolved codec.
    pub fn codec(&self) -> Codec {
        self.view.header().codec
    }

    /// Frames decoded so far.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Number of times the loop has wrapped.
    pub fn loop_count(&self) -> u32 {
        self.loop_count
    }

    /// Whether the stream is exhausted (and no loop applies).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Reset to the first data chunk, clearing predictor state.
    pub fn reset(&mut self) {
        self.next_chunk_offset = 0;
        self.started = false;
        self.finished = false;
        self.payload = &[];
        self.payload_pos = 0;
        self.in_payload_header = true;
        self.group_frames = 0;
        self.group_pos = 0;
        self.state = [AdpcmState::default(); MAX_CHANNELS];
        self.frames_decoded = 0;
        self.loop_count = 0;
    }

    fn loop_bounds(&self) -> Option<(u64, u64)> {
        let h = self.view.header();
        match (h.loop_start, h.loop_end) {
            (Some(s), Some(e)) if e > s => Some((u64::from(s), u64::from(e))),
            _ => None,
        }
    }

    /// Advance to the next data payload; returns `false` at end of stream.
    fn load_next_payload(&mut self) -> Result<bool> {
        loop {
            // Find next chunk starting at next_chunk_offset via linear scan
            // from that offset using the view's bytes.
            let bytes = self.view.bytes();
            if !self.started {
                self.started = true;
                self.next_chunk_offset = 0;
            }
            if self.next_chunk_offset >= bytes.len() {
                return Ok(false);
            }
            // Parse one chunk at next_chunk_offset.
            let mut iter_offset = self.next_chunk_offset;
            // Use a fresh ChunkIter positioned by scanning: cheaper to parse
            // the single chunk manually here.
            let chunk = parse_one(bytes, iter_offset)?;
            let chunk = match chunk {
                Some(c) => c,
                None => return Ok(false),
            };
            iter_offset = chunk.offset + chunk.size;
            self.next_chunk_offset = iter_offset;
            match chunk.kind {
                ChunkKind::Data => {
                    self.payload =
                        &bytes[chunk.payload_offset..chunk.payload_offset + chunk.payload_len];
                    self.payload_pos = 0;
                    self.in_payload_header = true;
                    return Ok(true);
                }
                ChunkKind::End => return Ok(false),
                _ => continue,
            }
        }
    }

    /// Refill the decoded group buffer. Returns false at EOF.
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
                // Skip the 8-byte SCDl sub-header.
                let skip = SCDL_HEADER_LEN.min(self.payload.len());
                self.payload_pos = skip;
                self.in_payload_header = false;
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
                    for f in 0..frames {
                        for ch in 0..channels {
                            let off = f * frame_bytes + ch * (frame_bytes / channels);
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
                    if avail.len() < 16 {
                        self.payload_pos = self.payload.len();
                        continue;
                    }
                    if channels == 1 {
                        let n = decode_spu_block(
                            &mut self.state[0],
                            &avail[..16],
                            &mut self.group[..28],
                        )?;
                        self.group_frames = n;
                    } else {
                        // Stereo: blocks alternate L, R. A pair yields 28 frames.
                        if avail.len() < 32 {
                            self.payload_pos = self.payload.len();
                            continue;
                        }
                        let mut l = [0i16; 28];
                        let mut r = [0i16; 28];
                        decode_spu_block(&mut self.state[0], &avail[..16], &mut l)?;
                        decode_spu_block(&mut self.state[1], &avail[16..32], &mut r)?;
                        for i in 0..28 {
                            self.group[i * 2] = l[i];
                            self.group[i * 2 + 1] = r[i];
                        }
                        self.group_frames = 28;
                        self.payload_pos += 16; // second block consumed below via extra advance
                                                // Consume the right block as well.
                        self.payload_pos += 16;
                        self.group_pos = 0;
                        return Ok(true);
                    }
                    self.payload_pos += 16;
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::EaXaV1
                | Codec::EaXaV2
                | Codec::EaAdpcmR1
                | Codec::EaAdpcmR2
                | Codec::EaAdpcmR3 => {
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
                        decode_ea_xa_subframe_at(
                            &mut self.state[0],
                            &avail[..subframe_len],
                            data_offset,
                            &mut self.group[..28],
                            high_first,
                        )?;
                        self.group_frames = 28;
                    } else {
                        let mut l = [0i16; 28];
                        let mut r = [0i16; 28];
                        decode_ea_xa_subframe_at(
                            &mut self.state[0],
                            &avail[..subframe_len],
                            data_offset,
                            &mut l,
                            high_first,
                        )?;
                        decode_ea_xa_subframe_at(
                            &mut self.state[1],
                            &avail[subframe_len..subframe_len * 2],
                            data_offset,
                            &mut r,
                            high_first,
                        )?;
                        for i in 0..28 {
                            self.group[i * 2] = l[i];
                            self.group[i * 2 + 1] = r[i];
                        }
                        self.group_frames = 28;
                    }
                    self.payload_pos += needed;
                    self.group_pos = 0;
                    return Ok(true);
                }
                Codec::Unknown(id) => return Err(Error::UnsupportedCodec(id)),
            }
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
            // Loop-end check before emitting more frames.
            if let Some((_s, e)) = self.loop_bounds() {
                if self.frames_decoded >= e {
                    self.seek_to_frame(u64::from(self.view.header().loop_start.unwrap_or(0)))?;
                    self.loop_count = self.loop_count.saturating_add(1);
                }
            }
            if self.group_pos >= self.group_frames {
                if self.finished {
                    break;
                }
                match self.refill_group()? {
                    true => {}
                    false => {
                        if let Some((s, _e)) = self.loop_bounds() {
                            if self.frames_decoded > s {
                                self.seek_to_frame(s)?;
                                self.loop_count = self.loop_count.saturating_add(1);
                                continue;
                            }
                        }
                        self.finished = true;
                        break;
                    }
                }
            }
            let avail_frames = self.group_frames - self.group_pos;
            let mut take = avail_frames.min(max_frames - written);
            // Respect loop_end boundary.
            if let Some((_s, e)) = self.loop_bounds() {
                let remaining = e.saturating_sub(self.frames_decoded) as usize;
                take = take.min(remaining);
                if take == 0 {
                    continue;
                }
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

    /// Decode into separate planar `f32` left/right buffers.
    /// For mono, the signal is written to `left` and `right` is filled
    /// with the same samples. Returns frames written.
    pub fn decode_f32_planar(&mut self, left: &mut [f32], right: &mut [f32]) -> Result<usize> {
        let n = left.len().min(right.len());
        // Decode via a fixed stack buffer in chunks.
        let mut buf = [0i16; 256 * MAX_CHANNELS];
        let mut done = 0usize;
        while done < n {
            let want = (n - done).min(256);
            let got = self.decode_i16(&mut buf[..want * self.channels() as usize])?;
            if got == 0 {
                break;
            }
            let ch = self.channels() as usize;
            for i in 0..got {
                let l = buf[i * ch];
                let r = if ch == 2 { buf[i * ch + 1] } else { l };
                left[done + i] = f32::from(l) / 32768.0;
                right[done + i] = f32::from(r) / 32768.0;
            }
            done += got;
            if got < want {
                break;
            }
        }
        Ok(done)
    }

    /// Seek by resetting and discarding decoded frames up to `target`.
    fn seek_to_frame(&mut self, target: u64) -> Result<()> {
        let saved_loop = self.loop_count;
        self.reset();
        self.loop_count = saved_loop;
        let discard = [0i16; 512 * MAX_CHANNELS];
        let mut remaining = target;
        // Temporarily clear loop bounds effect by decoding with raw refill:
        // decode_i16 would re-trigger looping, so discard via group refill.
        while remaining > 0 {
            if self.group_pos >= self.group_frames && !self.refill_group()? {
                break;
            }
            let avail = (self.group_frames - self.group_pos) as u64;
            let take = avail.min(remaining) as usize;
            self.group_pos += take;
            self.frames_decoded += take as u64;
            remaining -= take as u64;
            let _ = &discard;
        }
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
