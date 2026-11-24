//! Gated local verification against real PS1 `SCHl` streams.
//! Skips unless `EA_AUDIO_SAMPLE_DIR` points at a directory containing
//! `sample_01.bin` / `sample_02.bin`. No game bytes are stored in this repo.

use libea_audio::{EaDecoder, EaStreamView};

#[test]
fn real_ps1_schl_samples_decode_cleanly() {
    let Ok(dir) = std::env::var("EA_AUDIO_SAMPLE_DIR") else {
        eprintln!("EA_AUDIO_SAMPLE_DIR not set; skipping real-sample check");
        return;
    };
    for name in ["sample_01.bin", "sample_02.bin"] {
        let path = format!("{dir}/{name}");
        let bytes = std::fs::read(&path).expect("read sample");
        let view = EaStreamView::parse(&bytes).expect("parse SCHl");
        let expect_frames = u64::from(view.header().num_samples.expect("num_samples"));
        let channels = view.header().channels as usize;
        let mut dec = EaDecoder::new(view);
        let mut buf = [0i16; 8192];
        let mut total: u64 = 0;
        let mut clipped: u64 = 0;
        let mut sum_sq: f64 = 0.0;
        let mut count: u64 = 0;
        loop {
            let n = dec.decode_i16(&mut buf).unwrap() as u64;
            if n == 0 {
                break;
            }
            for &s in &buf[..(n as usize * channels)] {
                if s == i16::MAX || s == i16::MIN {
                    clipped += 1;
                }
                sum_sq += f64::from(s) * f64::from(s);
                count += 1;
            }
            total += n;
            if total > expect_frames {
                break;
            }
        }
        let rms = (sum_sq / count.max(1) as f64).sqrt();
        eprintln!(
            "{name}: frames={total} expect={expect_frames} clipped={clipped}/{count} rms={rms:.1}"
        );
        assert_eq!(total, expect_frames, "{name} frame count");
        assert!(rms > 100.0, "{name} should not be silent (rms {rms})");
        // Non-clipping: at most 0.1% of samples at the rails.
        assert!(
            clipped * 1000 <= count,
            "{name} clips excessively: {clipped}/{count}"
        );
    }
}
