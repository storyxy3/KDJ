//! Mixxx 式 RGB 波形：**每一列一根柱子**，高度 = 全带峰值，颜色 = 三分频合成。
//!
//! 对照 [mixxxdj/mixxx](https://github.com/mixxxdj/mixxx) 的 `AnalyzerWaveform`：
//! - 时域 IIR 三分频（Mixxx 用 Bessel4；这里用二阶 Butterworth 双二阶，交叉点同为
//!   **600 Hz / 4000 Hz**——见 `analyzerwaveform.cpp` 的 `kLowMidFreqHz` /
//!   `kMidHighFreqHz`）
//! - 每个显示格取 `|x|` **峰值**（不是 STFT 功率、也不是均值）
//! - 颜色与 `WaveformRendererRGB` 一致：`RGB = Σ band·primary`，再按最大分量归一
//!   （默认 low=红 mid=绿 high=蓝）
//!
//! 高度仍做 P5→P99 对比拉伸：Mixxx 是把 0..1 样本直接 ×255 存库、渲染时再乘
//! gain；我们的前端直接吃 0..1 的 `amp`，母带压扁的曲子不拉伸会变成一条实心带。
//!
//! 波形单开一条路径：纯展示，不影响 BPM/调性，也不逼用户为了看波形重跑分析。

use kdj_core::models::Waveform;

use crate::dsp::percentile;

/// 16 kHz：奈奎斯特 8 kHz，盖住 4 kHz 高通交叉点还有一倍余量；
/// 再高只是让解码变慢，对几百像素宽的总览波形没有意义。
pub const WAVEFORM_SR: u32 = 16000;

/// Mixxx `AnalyzerWaveform` 交叉点（`analyzerwaveform.cpp`）。
const XOVER_LOW_MID_HZ: f64 = 600.0;
const XOVER_MID_HIGH_HZ: f64 = 4000.0;

const AMP_GAMMA: f64 = 1.2;
/// 通道下限：纯 (255,0,0) 在深色底上太扎眼，抬一点让暗通道留一丝底色。
const COLOR_FLOOR: f64 = 0.12;

/// 稳定后再取峰值，跳过 IIR 起振（Mixxx 用 `assumeSettled()` 预热静音）。
const SETTLE_SECONDS: f64 = 0.05;

pub fn band_waveform(samples: &[f32], sr: f64, buckets: usize) -> Waveform {
    let buckets = buckets.clamp(64, 2000);
    if samples.len() < 32 || sr <= 0.0 {
        return Waveform::default();
    }

    let (all, low, mid, high) = filter_peak_buckets(samples, sr, buckets);
    let count = all.len();
    if count == 0 {
        return Waveform::default();
    }

    // ---- 高度：全带峰值 + 百分位对比拉伸（P5 地板 / P99 顶）
    let mut amp = all;
    let mut sorted = amp.clone();
    sorted.sort_by(f64::total_cmp);
    let hi = {
        let value = percentile(&sorted, 99.0);
        if value > 0.0 {
            value
        } else {
            1.0
        }
    };
    let lo = percentile(&sorted, 5.0);
    for value in amp.iter_mut() {
        *value = ((*value - lo) / (hi - lo).max(1e-9))
            .clamp(0.0, 1.0)
            .powf(AMP_GAMMA);
    }

    // ---- 颜色：Mixxx RGB = low·R + mid·G + high·B，再 / max(分量)
    // 默认 primary 就是单位矩阵，化简为直接用三段峰值当 RGB 再归一。
    // 与旧 Serato「相对本曲占比」不同：这里跨曲同一颜色更接近同一绝对频谱。
    let mut r = vec![0u8; count];
    let mut g = vec![0u8; count];
    let mut b = vec![0u8; count];
    for i in 0..count {
        let mut red = low[i];
        let mut green = mid[i];
        let mut blue = high[i];
        let peak = red.max(green).max(blue);
        if peak > 0.0 {
            red /= peak;
            green /= peak;
            blue /= peak;
        }
        r[i] = to_u8(red);
        g[i] = to_u8(green);
        b[i] = to_u8(blue);
    }

    Waveform {
        track_id: 0,
        duration: ((samples.len() as f64 / sr) * 1000.0).round() / 1000.0,
        amp: amp
            .into_iter()
            .map(|v| ((v * 10_000.0).round() / 10_000.0) as f32)
            .collect(),
        r,
        g,
        b,
    }
}

fn to_u8(channel: f64) -> u8 {
    let lifted = COLOR_FLOOR + (1.0 - COLOR_FLOOR) * channel.clamp(0.0, 1.0);
    (lifted * 255.0).round() as u8
}

/// 时域三分频后，按显示格取 |x| 峰值。返回 (all, low, mid, high)。
fn filter_peak_buckets(
    samples: &[f32],
    sr: f64,
    buckets: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let step = (n / buckets).max(1);
    let count = n / step;
    if count == 0 {
        return Default::default();
    }

    // 两级串联逼近 4 阶（Mixxx Bessel4）；可视化对相位不敏感，Butterworth 够用。
    let mut low_a = Biquad::lowpass(sr, XOVER_LOW_MID_HZ, 0.7071);
    let mut low_b = Biquad::lowpass(sr, XOVER_LOW_MID_HZ, 0.7071);
    let mut high_a = Biquad::highpass(sr, XOVER_MID_HIGH_HZ, 0.7071);
    let mut high_b = Biquad::highpass(sr, XOVER_MID_HIGH_HZ, 0.7071);
    // 带通 = 高通(低交叉) → 低通(高交叉)，与 Mixxx band 滤波器同交叉点。
    let mut mid_hp_a = Biquad::highpass(sr, XOVER_LOW_MID_HZ, 0.7071);
    let mut mid_hp_b = Biquad::highpass(sr, XOVER_LOW_MID_HZ, 0.7071);
    let mut mid_lp_a = Biquad::lowpass(sr, XOVER_MID_HIGH_HZ, 0.7071);
    let mut mid_lp_b = Biquad::lowpass(sr, XOVER_MID_HIGH_HZ, 0.7071);

    let settle = ((SETTLE_SECONDS * sr) as usize).min(n.saturating_sub(1));
    for sample in samples.iter().take(settle) {
        let x = f64::from(*sample);
        let _ = low_b.process(low_a.process(x));
        let _ = high_b.process(high_a.process(x));
        let _ = mid_lp_b.process(mid_lp_a.process(mid_hp_b.process(mid_hp_a.process(x))));
    }

    let mut all = vec![0.0f64; count];
    let mut low = vec![0.0f64; count];
    let mut mid = vec![0.0f64; count];
    let mut high = vec![0.0f64; count];

    for (index, slot_all) in all.iter_mut().enumerate() {
        let start = index * step;
        let end = start + step;
        let mut peak_all = 0.0f64;
        let mut peak_low = 0.0f64;
        let mut peak_mid = 0.0f64;
        let mut peak_high = 0.0f64;
        for sample in &samples[start..end] {
            let x = f64::from(*sample);
            let y_low = low_b.process(low_a.process(x));
            let y_high = high_b.process(high_a.process(x));
            let y_mid =
                mid_lp_b.process(mid_lp_a.process(mid_hp_b.process(mid_hp_a.process(x))));
            peak_all = peak_all.max(x.abs());
            peak_low = peak_low.max(y_low.abs());
            peak_mid = peak_mid.max(y_mid.abs());
            peak_high = peak_high.max(y_high.abs());
        }
        *slot_all = peak_all;
        low[index] = peak_low;
        mid[index] = peak_mid;
        high[index] = peak_high;
    }

    (all, low, mid, high)
}

/// Transposed Direct Form II 双二阶（RBJ cookbook）。
#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn lowpass(sr: f64, cutoff: f64, q: f64) -> Self {
        let w0 = std::f64::consts::TAU * (cutoff / sr).clamp(1e-6, 0.49);
        let cos = w0.cos();
        let sin = w0.sin();
        let alpha = sin / (2.0 * q.max(1e-6));
        let b0 = (1.0 - cos) * 0.5;
        let b1 = 1.0 - cos;
        let b2 = (1.0 - cos) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    fn highpass(sr: f64, cutoff: f64, q: f64) -> Self {
        let w0 = std::f64::consts::TAU * (cutoff / sr).clamp(1e-6, 0.49);
        let cos = w0.cos();
        let sin = w0.sin();
        let alpha = sin / (2.0 * q.max(1e-6));
        let b0 = (1.0 + cos) * 0.5;
        let b1 = -(1.0 + cos);
        let b2 = (1.0 + cos) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    fn normalize(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, seconds: f64, sr: f64) -> Vec<f32> {
        (0..(seconds * sr) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin() as f32)
            .collect()
    }

    #[test]
    fn every_channel_has_the_same_length_as_amp() {
        let samples = tone(440.0, 10.0, WAVEFORM_SR as f64);
        let wave = band_waveform(&samples, WAVEFORM_SR as f64, 200);
        assert!(!wave.amp.is_empty());
        assert_eq!(wave.r.len(), wave.amp.len());
        assert_eq!(wave.g.len(), wave.amp.len());
        assert_eq!(wave.b.len(), wave.amp.len());
        assert!((wave.duration - 10.0).abs() < 0.05);
    }

    #[test]
    fn bucket_count_follows_the_integer_division_rule() {
        // 分格是 `step = max(1, n_samples/buckets)` 再 `count = n_samples/step`。
        // 样本够多时实际列数接近请求值；不够时按能整除的列数给。
        let sr = WAVEFORM_SR as f64;
        let samples_of = |seconds: f64| (seconds * sr) as usize;

        let short = band_waveform(&tone(440.0, 1.0, sr), sr, 640);
        let expected = samples_of(1.0) / (samples_of(1.0) / 640).max(1);
        assert_eq!(short.amp.len(), expected);

        let long_samples = tone(440.0, 300.0, sr);
        for buckets in [100usize, 300, 640] {
            let wave = band_waveform(&long_samples, sr, buckets);
            assert!(
                wave.amp.len() >= buckets && wave.amp.len() <= buckets + 1,
                "请求 {buckets} 格，实际 {}",
                wave.amp.len()
            );
        }
    }

    #[test]
    fn bass_sections_read_redder_than_treble_sections_of_the_same_track() {
        // Mixxx 绝对频谱上色：100 Hz 应偏红，5 kHz 应偏蓝。
        let sr = WAVEFORM_SR as f64;
        let mut samples = tone(100.0, 8.0, sr);
        samples.extend(tone(5000.0, 8.0, sr));

        let wave = band_waveform(&samples, sr, 200);
        let half = wave.amp.len() / 2;
        let bass_at = half / 2;
        let treble_at = half + half / 2;

        assert!(
            wave.r[bass_at] > wave.b[bass_at],
            "低频段应当偏红：r={} b={}",
            wave.r[bass_at],
            wave.b[bass_at]
        );
        assert!(
            wave.b[treble_at] > wave.r[treble_at],
            "高频段应当偏蓝：r={} b={}",
            wave.r[treble_at],
            wave.b[treble_at]
        );
    }

    #[test]
    fn mid_tone_reads_greener_than_bass_or_treble() {
        let sr = WAVEFORM_SR as f64;
        // 1 kHz 落在 600–4000 中频带
        let wave = band_waveform(&tone(1000.0, 4.0, sr), sr, 100);
        let i = wave.amp.len() / 2;
        assert!(
            wave.g[i] > wave.r[i] && wave.g[i] > wave.b[i],
            "1 kHz 应偏绿：r={} g={} b={}",
            wave.r[i],
            wave.g[i],
            wave.b[i]
        );
    }

    #[test]
    fn amplitudes_stay_inside_the_unit_range() {
        let samples = tone(440.0, 10.0, WAVEFORM_SR as f64);
        let wave = band_waveform(&samples, WAVEFORM_SR as f64, 200);
        assert!(wave.amp.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn colour_channels_never_go_fully_black() {
        let samples = tone(100.0, 10.0, WAVEFORM_SR as f64);
        let wave = band_waveform(&samples, WAVEFORM_SR as f64, 200);
        let floor = (COLOR_FLOOR * 255.0).round() as u8;
        assert!(wave.r.iter().all(|v| *v >= floor));
        assert!(wave.g.iter().all(|v| *v >= floor));
        assert!(wave.b.iter().all(|v| *v >= floor));
    }

    #[test]
    fn too_short_input_returns_an_empty_waveform_instead_of_panicking() {
        let wave = band_waveform(&[0.0; 16], WAVEFORM_SR as f64, 200);
        assert!(wave.amp.is_empty());
    }
}
