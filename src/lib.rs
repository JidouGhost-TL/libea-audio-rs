//! Standalone pure-Rust EA `SCHl` / ASF / EA-XA / EACS audio codec and
//! stream decoder with zero runtime dependencies.
//!
//! Chunk framing, PT TLV headers, EA-XA (v1/v2), PS1 SPU-ADPCM payloads
//! inside PS1 `SCHl`, PCM8/PCM16 (LE & BE) and EA-ADPCM R1/R2/R3 are
//! supported. Hot decode loops perform no heap allocation; the
//! [`EaStreamView`] / [`EaDecoder`] streaming API is zero-allocation.
//!
//! EA ASF files use the same `SCHl` chunk framing as EACS streams, so
//! [`EaStreamView::parse`] handles both.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod codec;
pub mod decoder;

pub use codec::{
    decode_ea_xa_subframe, decode_ea_xa_subframe_at, decode_spu_block, ea_xa_step, spu_step,
    AdpcmState, Codec, EA_XA_PREDICTORS, SPU_PREDICTORS,
};
pub use decoder::EaDecoder;

use core::fmt;

/// Result type for this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Error returned by parsing and decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input ended before a required field completed.
    Truncated {
        /// Byte offset where truncation was detected.
        offset: usize,
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        available: usize,
    },
    /// Stream did not begin with the expected magic.
    BadMagic {
        /// Byte offset of the magic.
        offset: usize,
        /// Expected magic bytes.
        expected: [u8; 4],
        /// Found magic bytes.
        found: [u8; 4],
    },
    /// A chunk declared a size smaller than its 8-byte header.
    InvalidChunkSize {
        /// Byte offset of the chunk.
        offset: usize,
        /// Declared chunk size.
        size: u32,
    },
    /// The leading `SCHl`/`SCHb` chunk contained no `PT\x01\x00` header.
    MissingPtHeader {
        /// Byte offset of the header chunk.
        offset: usize,
    },
    /// A PT TLV record was malformed.
    MalformedPtTlv {
        /// Byte offset of the TLV record.
        offset: usize,
        /// TLV tag byte.
        tag: u8,
        /// Static reason string.
        reason: &'static str,
    },
    /// Codec id from the PT header is not implemented.
    UnsupportedCodec(u32),
    /// Channel count was zero or above the supported maximum (2).
    InvalidChannels(u32),
    /// ADPCM data ended mid-frame/block.
    TruncatedData {
        /// Bytes needed to complete the frame.
        needed: usize,
        /// Bytes available.
        available: usize,
    },
    /// Caller-provided output buffer is too small.
    OutputTooSmall {
        /// Samples needed.
        needed: usize,
        /// Samples available.
        available: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                offset,
                needed,
                available,
            } => {
                write!(
                    f,
                    "truncated at {offset:#x}: needed {needed}, available {available}"
                )
            }
            Self::BadMagic {
                offset,
                expected,
                found,
            } => {
                write!(
                    f,
                    "bad magic at {offset:#x}: expected {expected:?}, found {found:?}"
                )
            }
            Self::InvalidChunkSize { offset, size } => {
                write!(f, "invalid chunk size {size} at {offset:#x}")
            }
            Self::MissingPtHeader { offset } => write!(f, "missing PT header at {offset:#x}"),
            Self::MalformedPtTlv {
                offset,
                tag,
                reason,
            } => {
                write!(
                    f,
                    "malformed PT TLV tag {tag:#04x} at {offset:#x}: {reason}"
                )
            }
            Self::UnsupportedCodec(id) => write!(f, "unsupported codec id {id}"),
            Self::InvalidChannels(n) => write!(f, "invalid channel count {n}"),
            Self::TruncatedData { needed, available } => {
                write!(f, "truncated data: needed {needed}, available {available}")
            }
            Self::OutputTooSmall { needed, available } => {
                write!(
                    f,
                    "output too small: needed {needed}, available {available}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// `SCHl` header chunk tag (little-endian family).
pub const MAGIC_SCHL: [u8; 4] = *b"SCHl";
/// `SCCl` chunk-count tag.
pub const MAGIC_SCCL: [u8; 4] = *b"SCCl";
/// `SCDl` audio data tag.
pub const MAGIC_SCDL: [u8; 4] = *b"SCDl";
/// `SCLl` loop tag.
pub const MAGIC_SCLL: [u8; 4] = *b"SCLl";
/// `SCEl` end tag.
pub const MAGIC_SCEL: [u8; 4] = *b"SCEl";
/// `SCHb` header chunk tag (big-endian family).
pub const MAGIC_SCHB: [u8; 4] = *b"SCHb";
/// `SCDb` audio data tag (big-endian family).
pub const MAGIC_SCDB: [u8; 4] = *b"SCDb";
/// `SCEb` end tag (big-endian family).
pub const MAGIC_SCEB: [u8; 4] = *b"SCEb";
/// PT header signature.
pub const PT_MAGIC: [u8; 4] = *b"PT\x01\x00";
/// Default sample rate when PT tag `0x84` is absent.
pub const DEFAULT_SAMPLE_RATE: u32 = 22_050;
/// Default channel count when PT tag `0x82` is absent.
pub const DEFAULT_CHANNELS: u32 = 1;

const CHUNK_HEADER_LEN: usize = 8;
/// Bytes of sub-header at the start of every `SCDl` payload in SCHl streams.
pub const SCDL_HEADER_LEN: usize = 8;

/// Classification of a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// `SCHl` / `SCHb` header.
    Header,
    /// `SCCl` count metadata.
    Count,
    /// `SCDl` / `SCDb` audio data.
    Data,
    /// `SCLl` loop marker.
    Loop,
    /// `SCEl` / `SCEb` end marker.
    End,
    /// Any other tag.
    Other([u8; 4]),
}

impl ChunkKind {
    fn from_tag(tag: [u8; 4]) -> Self {
        match &tag {
            b"SCHl" | b"SCHb" => Self::Header,
            b"SCCl" | b"SCCb" => Self::Count,
            b"SCDl" | b"SCDb" => Self::Data,
            b"SCLl" | b"SCLb" => Self::Loop,
            b"SCEl" | b"SCEb" => Self::End,
            _ => Self::Other(tag),
        }
    }
    fn is_big_endian(tag: [u8; 4]) -> bool {
        tag[3] == b'b'
    }
}

/// One parsed chunk header and payload span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Classified kind.
    pub kind: ChunkKind,
    /// Raw tag bytes.
    pub tag: [u8; 4],
    /// Offset of the 8-byte chunk header.
    pub offset: usize,
    /// Total chunk length including header.
    pub size: usize,
    /// Offset of the payload.
    pub payload_offset: usize,
    /// Payload length.
    pub payload_len: usize,
    /// Chunk size field is big-endian.
    pub big_endian: bool,
}

/// Parsed PT TLV header fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Optional PS1 stream word at `+0x08` (PT at `+0x0c`).
    pub ps1_stream_word: Option<u32>,
    /// Absolute offset of the PT magic.
    pub pt_offset: usize,
    /// Channel count (1 or 2).
    pub channels: u32,
    /// Raw codec id from tag `0x83`, if present.
    pub codec_id: Option<u32>,
    /// Resolved codec (inferred when `codec_id` is absent).
    pub codec: Codec,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total per-channel sample count from tag `0x85`, if present.
    pub num_samples: Option<u32>,
    /// Loop start sample from tag `0x86`, if present.
    pub loop_start: Option<u32>,
    /// Loop end sample from tag `0x87`, if present.
    pub loop_end: Option<u32>,
    /// Stream flags from tag `0x8a`, if present.
    pub flags: Option<u32>,
}

fn read_chunk_size(bytes: &[u8], offset: usize, be: bool) -> Result<usize> {
    let s = bytes.get(offset..offset + 4).ok_or(Error::Truncated {
        offset,
        needed: 4,
        available: bytes.len().saturating_sub(offset),
    })?;
    let v = if be {
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    } else {
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    };
    if (v as usize) < CHUNK_HEADER_LEN {
        return Err(Error::InvalidChunkSize {
            offset: offset - 4,
            size: v,
        });
    }
    Ok(v as usize)
}

fn parse_pt(chunk: &[u8], chunk_offset: usize) -> Result<Header> {
    let payload = chunk.get(CHUNK_HEADER_LEN..).ok_or(Error::Truncated {
        offset: chunk_offset,
        needed: CHUNK_HEADER_LEN,
        available: chunk.len(),
    })?;
    let (ps1_stream_word, pt_rel) = if payload.get(0..4) == Some(&PT_MAGIC[..]) {
        (None, CHUNK_HEADER_LEN)
    } else if payload.len() >= 8 && payload.get(4..8) == Some(&PT_MAGIC[..]) {
        (
            Some(u32::from_le_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ])),
            CHUNK_HEADER_LEN + 4,
        )
    } else {
        return Err(Error::MissingPtHeader {
            offset: chunk_offset,
        });
    };
    let mut h = Header {
        ps1_stream_word,
        pt_offset: chunk_offset + pt_rel,
        channels: DEFAULT_CHANNELS,
        codec_id: None,
        codec: Codec::EaXaV2,
        sample_rate: DEFAULT_SAMPLE_RATE,
        num_samples: None,
        loop_start: None,
        loop_end: None,
        flags: None,
    };
    let mut cursor = pt_rel + 4;
    while cursor < chunk.len() {
        let tag_offset = chunk_offset + cursor;
        let tag = chunk[cursor];
        cursor += 1;
        match tag {
            // 0xFC pad, 0xFD/0xFE section markers.
            0xFC..=0xFE => continue,
            0xFF => break,
            _ => {
                let len_byte = *chunk.get(cursor).ok_or(Error::Truncated {
                    offset: chunk_offset + cursor,
                    needed: 1,
                    available: 0,
                })?;
                cursor += 1;
                let val_len = usize::from(len_byte);
                if val_len > 4 {
                    return Err(Error::MalformedPtTlv {
                        offset: tag_offset,
                        tag,
                        reason: "integer value exceeds 4 bytes",
                    });
                }
                let end = cursor + val_len;
                let val = chunk.get(cursor..end).ok_or(Error::Truncated {
                    offset: chunk_offset + cursor,
                    needed: val_len,
                    available: chunk.len().saturating_sub(cursor),
                })?;
                cursor = end;
                let mut v: u32 = 0;
                for &b in val {
                    v = (v << 8) | u32::from(b);
                }
                match tag {
                    0x82 => h.channels = v,
                    0x83 => h.codec_id = Some(v),
                    0x84 => h.sample_rate = v,
                    0x85 => h.num_samples = Some(v),
                    0x86 => h.loop_start = Some(v),
                    0x87 => h.loop_end = Some(v),
                    0x8A => h.flags = Some(v),
                    _ => {}
                }
            }
        }
    }
    if h.channels == 0 || h.channels > 2 {
        return Err(Error::InvalidChannels(h.channels));
    }
    h.codec = match h.codec_id {
        Some(id) => {
            let c = Codec::from_pt_id(id);
            if let Codec::Unknown(x) = c {
                return Err(Error::UnsupportedCodec(x));
            }
            c
        }
        None if h.ps1_stream_word.is_some() => Codec::Ps1SpuAdpcm,
        None => Codec::EaXaV2,
    };
    Ok(h)
}

/// Zero-allocation view over an `SCHl`/ASF byte stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EaStreamView<'a> {
    bytes: &'a [u8],
    header: Header,
    first_chunk_size: usize,
    big_endian: bool,
}

impl<'a> EaStreamView<'a> {
    /// Parse and validate the leading header chunk. Chunk iteration is lazy.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < CHUNK_HEADER_LEN {
            return Err(Error::Truncated {
                offset: 0,
                needed: CHUNK_HEADER_LEN,
                available: bytes.len(),
            });
        }
        let tag = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if tag != MAGIC_SCHL && tag != MAGIC_SCHB {
            return Err(Error::BadMagic {
                offset: 0,
                expected: MAGIC_SCHL,
                found: tag,
            });
        }
        let be = ChunkKind::is_big_endian(tag);
        let size = read_chunk_size(bytes, 4, be)?;
        if size > bytes.len() {
            return Err(Error::Truncated {
                offset: 0,
                needed: size,
                available: bytes.len(),
            });
        }
        let header = parse_pt(&bytes[..size], 0)?;
        Ok(Self {
            bytes,
            header,
            first_chunk_size: size,
            big_endian: be,
        })
    }

    /// Return `true` if `bytes` look like an `SCHl` stream.
    pub fn identifies(bytes: &[u8]) -> bool {
        EaStreamView::parse(bytes).is_ok()
    }

    /// Parsed PT header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Full source bytes.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Whether the stream uses the big-endian chunk family.
    pub fn is_big_endian(&self) -> bool {
        self.big_endian
    }

    /// Size of the leading header chunk.
    pub fn first_chunk_size(&self) -> usize {
        self.first_chunk_size
    }

    /// Iterate chunks lazily; iteration stops after the end chunk or on error.
    pub fn chunks(&self) -> ChunkIter<'a> {
        ChunkIter {
            bytes: self.bytes,
            offset: 0,
            done: false,
        }
    }

    /// Payload slices of all `SCDl`/`SCDb` chunks (including 8-byte sub-header).
    pub fn data_payloads(&self) -> impl Iterator<Item = Result<&'a [u8]>> + 'a {
        let bytes = self.bytes;
        self.chunks().filter_map(move |c| match c {
            Ok(ch) if ch.kind == ChunkKind::Data => Some(Ok(
                &bytes[ch.payload_offset..ch.payload_offset + ch.payload_len]
            )),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
    }
}

/// Lazy chunk iterator over a stream.
#[derive(Debug, Clone)]
pub struct ChunkIter<'a> {
    bytes: &'a [u8],
    offset: usize,
    done: bool,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = Result<Chunk>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.offset >= self.bytes.len() {
            return None;
        }
        let offset = self.offset;
        let remaining = self.bytes.len() - offset;
        if remaining < CHUNK_HEADER_LEN {
            self.done = true;
            return Some(Err(Error::Truncated {
                offset,
                needed: CHUNK_HEADER_LEN,
                available: remaining,
            }));
        }
        let tag = [
            self.bytes[offset],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        ];
        let be = ChunkKind::is_big_endian(tag);
        let size = match read_chunk_size(self.bytes, offset + 4, be) {
            Ok(s) => s,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        if offset + size > self.bytes.len() {
            self.done = true;
            return Some(Err(Error::Truncated {
                offset,
                needed: size,
                available: remaining,
            }));
        }
        let kind = ChunkKind::from_tag(tag);
        let chunk = Chunk {
            kind,
            tag,
            offset,
            size,
            payload_offset: offset + CHUNK_HEADER_LEN,
            payload_len: size - CHUNK_HEADER_LEN,
            big_endian: be,
        };
        self.offset = offset + size;
        if kind == ChunkKind::End {
            self.done = true;
        }
        Some(Ok(chunk))
    }
}

/// Decode an entire stream into an interleaved `i16` vector (alloc only).
#[cfg(feature = "alloc")]
pub fn decode_all_i16(view: EaStreamView<'_>) -> Result<alloc::vec::Vec<i16>> {
    let mut dec = EaDecoder::new(view);
    let mut out = alloc::vec::Vec::new();
    let mut buf = [0i16; 1024];
    loop {
        let frames = dec.decode_i16(&mut buf)?;
        if frames == 0 {
            break;
        }
        let ch = dec.channels() as usize;
        out.extend_from_slice(&buf[..frames * ch]);
        if frames * ch < buf.len() && dec.is_finished() {
            break;
        }
    }
    Ok(out)
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    #[test]
    fn error_display_works() {
        assert!(format!("{}", Error::UnsupportedCodec(99)).contains("99"));
    }
}
