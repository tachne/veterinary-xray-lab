"""Veterinary radiography teaching prototype — phase 1 only, no clinical advice."""

from dataclasses import asdict, replace
from pathlib import Path
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from simulator import APP_VERSION
from simulator.comparison import MAX_SNAPSHOTS, MODEL_VERSION, make_snapshot, snapshots_csv
from simulator.imaging import make_phantom, sample_image
from simulator.patterns import PATTERN_SPECS, LINE_PAIR_GROUPS, make_extended_phantom, sample_extended_image
from simulator.physics import ExposureSettings, ModelParameters, calculate_exposure
from simulator.presentation import ENHANCEMENT_FACTORS, ZOOM_FACTORS, prepare_display

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Vet X-ray Lab | 撮影条件・画質評価", page_icon="◉", layout="wide")

st.markdown("""
<style>
.block-container {max-width:1440px; padding-top:4.2rem; padding-bottom:2rem;}
h1 {font-size:2rem!important; letter-spacing:-.04em!important;}
h2 {font-size:1.3rem!important;}
h3 {font-size:1.08rem!important;}
[data-testid="stMetric"] {background:white; border:1px solid #dce3ea; border-radius:12px; padding:14px; min-height:134px;}
[data-testid="stMetricLabel"] {font-size:.82rem;}
[data-testid="stMetricValue"] {font-size:clamp(1.1rem,2.2vw,1.7rem)!important;}
[data-testid="stMetric"] .stMarkdownEllipsis,
[data-testid="stMetric"] .stMarkdownEllipsis p,
[data-testid="stButton"] .stMarkdownEllipsis,
[data-testid="stButton"] .stMarkdownEllipsis p {white-space:normal!important; overflow:visible!important; text-overflow:clip!important;}
.eyebrow {font-size:12px; color:#087e8b; font-weight:750; letter-spacing:.16em; margin-bottom:6px;}
.subtle {color:#536779; font-size:14px; line-height:1.8;}
[data-testid="stSidebar"] {border-right:1px solid #dce3ea;}
[data-testid="stTabs"] button {font-weight:650;}
[data-testid="stImage"] img[src^="data:image/png"] {image-rendering:pixelated;}
</style>
""", unsafe_allow_html=True)

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
    st.session_state.next_snapshot_id = 1
if "noise_seed" not in st.session_state:
    st.session_state.noise_seed = 42


def regenerate_noise():
    st.session_state.noise_seed = (st.session_state.noise_seed + 1) % (2**32)
    st.session_state.fixed_seed = st.session_state.noise_seed


def apply_fixed_seed():
    st.session_state.noise_seed = int(st.session_state.fixed_seed)


with st.sidebar:
    st.markdown('<div class="eyebrow">VET X-RAY LAB</div>', unsafe_allow_html=True)
    st.header("撮影条件")
    st.caption("第1段階 · 近似モデル")
    kvp = st.slider("管電圧 kVp", 40.0, 120.0, 80.0, 1.0, key="kvp",
                    help="出力と実効減弱係数の両方が変化します。線質やコントラストの変化の実測再現ではありません。")
    mas = st.slider("管電流時間積 mAs", 0.1, 20.0, 5.0, 0.1, key="mas")
    sid = st.slider("SID（cm）", 50.0, 200.0, 100.0, 5.0, key="sid",
                    help="焦点から検出器までの距離。入射量の逆二乗則のみを扱い、拡大率や幾何学的不鮮鋭は再現しません。")
    thickness = st.slider("体厚（cm）", 0.0, 35.0, 15.0, 0.5, key="thickness",
                          help="均一な仮想物質の厚さ。動物種、撮影部位、組織組成は区別しません。")
    st.divider()
    target = st.number_input("目標 EI（EIₜ・仮定値）", min_value=1.0, max_value=5000.0,
                             value=250.0, step=10.0, key="target_ei",
                             help="仮定値です。施設・装置・部位ごとの実測で定める臨床目標値ではありません。DIの基準だけが変わります。")
    with st.expander("仮定したモデル係数"):
        st.caption("すべて教育用の仮定値です。係数の変更は校正・検証を意味しません。")
        output = st.number_input("基準出力（µGy/mAs）", min_value=1.0, max_value=100.0, value=50.0, step=1.0,
                                 key="output", help="80 kVp・SID 100 cm・被写体なしの仮想出力。実測値ではありません。")
        exponent = st.number_input("出力の kVp 指数 n", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key="n")
        mu = st.number_input("80 kVp の実効減弱係数（cm⁻¹）", min_value=0.01, max_value=0.30, value=0.18,
                             step=0.01, format="%.2f", key="mu")
        mu_power = st.number_input("減弱係数の kVp 指数 p", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="p")
        grid = st.number_input("仮想グリッド透過率", min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                               key="grid", help="1.0 は透過損失なし。散乱線除去やコントラスト改善はモデル化しません。")
        quanta = st.number_input("仮想計数換算係数（個/画素/µGy）", min_value=100.0, max_value=10000.0,
                                 value=1000.0, step=100.0, key="quanta",
                                 help="単位信号領域の1画素当たりの仮想計数。DQEや実検出器の量子効率を表しません。")
    st.divider()
    st.subheader("画像表示")
    test_pattern = st.selectbox("共通テストパターン", ["extended", "legacy"], key="test_pattern",
        format_func=lambda value: "新：ポメラニアン・線対" if value == "extended" else "従来：丸・四角・スリット")
    display_gain = st.selectbox("ノイズ表示", ENHANCEMENT_FACTORS, key="display_gain",
        format_func=lambda value: "標準表示（1倍）" if value == 1 else f"教育用{value}倍強調表示")
    display_zoom = st.selectbox("共通拡大", ZOOM_FACTORS, key="display_zoom",
        format_func=lambda value: "全体（1倍）" if value == 1 else f"中央{value}倍（共通領域）")
    seed_mode = st.radio("乱数シードの扱い", ["fixed", "new_on_conditions"], key="seed_mode",
        format_func=lambda value: "固定" if value == "fixed" else "条件変更ごとに別標本")
    fixed_seed = st.number_input("固定シード", min_value=0, max_value=2**32 - 1, value=42, step=1,
        key="fixed_seed", disabled=seed_mode != "fixed", on_change=apply_fixed_seed)
    st.caption("強調倍率・拡大率・目標EIの変更では標本は変わりません。パターン切替は同じseedで再生成します。「ノイズを再生成」は現在の標本を更新します。")
    st.caption("実測データ：未使用\n\n装置固有の校正・妥当性検証：未実施")

pattern_size = 512 if test_pattern == "extended" else 256
ideal_signal = make_extended_phantom() if test_pattern == "extended" else make_phantom()


def sample_selected_pattern(mean_quanta, seed):
    """Use one shared expected signal for every current comparison pane."""
    if test_pattern == "extended":
        return sample_extended_image(mean_quanta, seed)
    return sample_image(mean_quanta, seed)

try:
    settings = ExposureSettings(kvp=kvp, mas=mas, sid_cm=sid, thickness_cm=thickness)
    parameters = ModelParameters(output_ugy_per_mas_at_reference=output, kvp_exponent=exponent,
                                 mu_ref_per_cm=mu, mu_kvp_exponent=mu_power, grid_transmission=grid,
                                 target_ei=target, quanta_per_ugy=quanta)
    result = calculate_exposure(settings, parameters)
except ValueError as exc:
    st.error(f"計算できない条件です：{exc}")
    st.stop()


# The seed advances only for image-producing physical input changes, not reruns.
noise_conditions = tuple(asdict(settings).values()) + tuple(
    value for key, value in asdict(parameters).items() if key != "target_ei"
)
if seed_mode == "fixed" and st.session_state.get("_last_seed_mode", seed_mode) != seed_mode:
    st.session_state.noise_seed = int(fixed_seed)
if (seed_mode == "new_on_conditions"
        and st.session_state.get("_last_noise_conditions", noise_conditions) != noise_conditions):
    st.session_state.noise_seed = (st.session_state.noise_seed + 1) % (2**32)
st.session_state._last_noise_conditions = noise_conditions
st.session_state._last_seed_mode = seed_mode

if "reference_snapshot" not in st.session_state:
    st.session_state.reference_snapshot = make_snapshot(0, "比較基準", settings, parameters,
        st.session_state.noise_seed, enhancement_factor=display_gain, zoom=display_zoom,
        seed_mode=seed_mode, image_pattern=test_pattern)


def show_simulated_image(mean_quanta, seed, caption, factor=None):
    """Use the same ROI, window and rendering for all compared images."""
    factor = display_gain if factor is None else factor
    try:
        raw = sample_selected_pattern(mean_quanta, seed)
    except ValueError:
        st.warning("この条件の仮想計数は画像生成の計算範囲（10⁻⁹～10¹² 個/画素）外です。模擬画像は表示せず、計算値と曲線を確認してください。値の切り上げは行っていません。")
    else:
        preview = prepare_display(raw, ideal_signal, factor=factor, zoom=display_zoom)
        mode = "標準表示（1倍）" if factor == 1 else f"教育用{factor}倍強調表示・実撮影の再現ではありません"
        st.image(preview.uri, width="stretch", caption=f"{caption} · {mode} · seed {seed}",
                 alt=f"{caption}：{mode}、seed {seed}、共通拡大{display_zoom}倍")
        y0, y1, x0, x1 = preview.crop_bounds
        st.caption(f"表示範囲 x[{x0}:{x1}] / y[{y0}:{y1}] · 固定窓 0–1.2 · 範囲外クリップ {100*preview.clipping_fraction:.2f}%")
        if preview.clipping_fraction > 0:
            st.warning("表示範囲外の画素を白黒にクリップしています。表示画素のSDが強調倍率に比例するとは限りません。")

st.markdown(f'<div class="eyebrow">EXPOSURE / IMAGE QUALITY · v{APP_VERSION}</div>', unsafe_allow_html=True)
st.title("撮影条件・画像評価")
st.markdown('<p class="subtle">獣医X線シミュレーター ｜ 撮影条件から検出器側のカーマ・EI・量子ノイズを計算します。</p>', unsafe_allow_html=True)
st.info("物理近似と仮定値による計算です。実測値は使用していません。EIは患者被ばく線量ではありません。", icon="ℹ️")

metrics = st.columns(4)
metrics[0].metric("検出器側 K〈近似〉", f"{result.detector_kerma_ugy:.4g} µGy",
                  help="被写体・仮想グリッド通過後の検出器位置での値。患者入射表面線量・臓器線量ではありません。")
metrics[1].metric("EI〈校正関係を仮定〉", f"{result.ei:.4g}", help="EI = 100 × K[µGy] を仮定した無次元値。装置の実測EIではありません。")
metrics[2].metric("DI〈目標との差〉", f"{result.di:+.2f}", help="10 log₁₀(EI / EIₜ)。診断可否や患者線量の判定値ではありません。")
metrics[3].metric("相対量子ノイズ〈理論〉", f"{100 * result.relative_noise:.4g} %",
                  help="信号1の均一領域で標準偏差/平均 = 1/√N。低信号領域では局所相対ノイズが大きくなります。")
st.caption(f"信号1の一様な基準領域：平均仮想計数 N = {result.mean_quanta:.4g} 個/画素 · 理論SNR = {result.snr:.4g} · 目標 EIₜ = {target:g}（仮定）")
st.caption("DI = 0 は設定した目標EIとの一致を示し、診断画質の合否を示しません。")
if display_gain > 1:
    st.warning(f"教育用{display_gain}倍強調表示：視覚的強調であり、実際の撮影画像を再現したものではありません。数値欄のK・EI・DI・理論ノイズ・SNRは強調前の物理計算値です。")

name_col, save_col, seed_col = st.columns([2.5, 1.2, 1.2], vertical_alignment="bottom")
with name_col:
    case_name = st.text_input("比較用の記録名", max_chars=50, placeholder="例：mAsを2倍", key="case_name")
with save_col:
    if st.button("＋ 現在の条件を記録", key="save_case", type="primary", width="stretch",
                 disabled=len(st.session_state.snapshots) >= MAX_SNAPSHOTS):
        snapshot = make_snapshot(st.session_state.next_snapshot_id, case_name, settings, parameters,
                                 st.session_state.noise_seed, enhancement_factor=display_gain,
                                 zoom=display_zoom, seed_mode=seed_mode, image_pattern=test_pattern)
        st.session_state.snapshots.append(snapshot)
        st.session_state.next_snapshot_id += 1
        st.session_state.image_cases = [s["id"] for s in st.session_state.snapshots[-2:]]
        st.toast(f"{snapshot['name']} を記録しました")
        st.rerun()
with seed_col:
    st.button("ノイズを再生成", key="resample", width="stretch", on_click=regenerate_noise)
if len(st.session_state.snapshots) >= MAX_SNAPSHOTS:
    st.caption("記録は8件までです。「条件を比較」から不要な記録を削除できます。")

image_tab, curve_tab, compare_tab, model_tab = st.tabs(["模擬画像", "応答曲線", f"条件を比較（{len(st.session_state.snapshots)}）", "計算モデル・仕様"])

with image_tab:
    st.subheader("規格化信号と量子ノイズ")
    st.info("この版は不鮮鋭（ぼけ）をシミュレーションしていません。粒状性と鮮鋭度は異なる特性です。ノイズで線や毛並みが見えにくくなることを、MTFや実際の空間分解能の低下として扱わないでください。")
    st.caption("模擬信号を固定して計数だけを変えます。kVpによる被写体コントラストの変化、SIDによる拡大、MTFは再現していません。全比較画像に選択中の同一パターンを使います。")
    if test_pattern == "extended":
        st.caption("512×512画素：左上＝従来パターン、右上＝ポメラニアン、下段＝線対6群。線幅／間隔は上段左から8/8・6/6・4/4画素、下段左から3/3・2/2・1/1画素です。")
        with st.expander("テストパターンの寸法・観察上の注意"):
            st.table(pd.DataFrame([
                {"位置": f"{'上段' if i < 3 else '下段'}・{['左', '中', '右'][i % 3]}",
                 "線幅（画素）": group["line_width_px"], "間隔（画素）": group["gap_width_px"],
                 "周期（画素）": group["line_width_px"] + group["gap_width_px"]}
                for i, group in enumerate(LINE_PAIR_GROUPS)
            ]))
            st.caption("寸法は合成画像の画素単位です。画素間隔mmやlp/mmへ換算せず、実測MTF・検出器の分解能を示すものではありません。犬は被毛・輪郭・濃淡を比較する模式パターンで、X線解剖像ではありません。")
            st.caption("ブラウザーの縮小表示では細い線が欠けたり見え方が変わる場合があります。全体表示と画像右上の全画面表示で確認してください。中央拡大は同一領域の切り出し・最近傍表示であり、ぼけ処理ではありません。中央拡大では周辺の線対・犬の一部が画面外になります。")
    comparison_mode = st.radio("画像の比較", ["standard_emphasis", "reference_current"],
        format_func=lambda value: "標準と強調（同一条件）" if value == "standard_emphasis" else "条件変更前後（基準と現在）",
        horizontal=True, key="image_comparison")
    if st.button("現在の条件を比較基準に固定", key="pin_reference"):
        st.session_state.reference_snapshot = make_snapshot(0, "比較基準", settings, parameters,
            st.session_state.noise_seed, enhancement_factor=display_gain, zoom=display_zoom,
            seed_mode=seed_mode, image_pattern=test_pattern)
    reference = st.session_state.reference_snapshot
    st.caption(f"共通の表示条件：固定窓 0–1.2、同一階調、中央{display_zoom}倍（1倍は全体）、平滑化なし。画像右上から全画面表示できます。現在のseed：{st.session_state.noise_seed}")
    if test_pattern == "extended":
        st.caption("最細線は1画素幅です。画面上の縮小で線の欠落・見かけの縞が生じる場合があります。全画面表示で確認し、表示縮小の影響をぼけやノイズと区別してください。")
    image_cols = st.columns(2, gap="medium")
    with image_cols[0]:
        if comparison_mode == "standard_emphasis":
            st.markdown("**元のPoisson画像 · 標準表示**")
            show_simulated_image(result.mean_quanta, st.session_state.noise_seed, "現在の条件", factor=1)
        else:
            st.markdown("**変更前 · 固定した比較基準**")
            s, r = reference["settings"], reference["result"]
            st.caption(f"{s['kvp']:g} kVp · {s['mas']:g} mAs · SID {s['sid_cm']:g} cm · 体厚 {s['thickness_cm']:g} cm")
            show_simulated_image(r["mean_quanta"], reference["seed"], "比較基準")
            st.caption(f"K {r['detector_kerma_ugy']:.4g} µGy · EI {r['ei']:.4g} · 理論相対ノイズ {100*r['relative_noise']:.4g}%")
    with image_cols[1]:
        st.markdown("**現在の条件 · 選択した表示**")
        if comparison_mode == "reference_current":
            st.caption(f"{kvp:g} kVp · {mas:g} mAs · SID {sid:g} cm · 体厚 {thickness:g} cm")
        show_simulated_image(result.mean_quanta, st.session_state.noise_seed,
                             "現在の条件")
        if comparison_mode == "reference_current":
            st.caption(f"K {result.detector_kerma_ugy:.4g} µGy · EI {result.ei:.4g} · 理論相対ノイズ {100*result.relative_noise:.4g}%")
    if comparison_mode == "reference_current":
        st.caption(f"検出器側Kの比（現在/基準）：{result.detector_kerma_ugy/reference['result']['detector_kerma_ugy']:.4g}。患者線量の比ではありません。左右には同じ強調倍率を適用しています。")
        if reference["seed"] != st.session_state.noise_seed:
            st.info("左右のseedが異なるため、条件の違いに加えて標本の違いも含まれます。比較基準の標本は固定されています。")
    st.caption("同じ条件・seed・パターン版・NumPy版で再現できます。パターン切替では計算値は不変ですが、同じseedでも標本の配置は変わります。同じseedでも計数Nが変わると、画素ごとのノイズ模様が完全に対応するわけではありません。")
    st.caption("画像は仮想計数Cを基準の平均計数Nで割った規格化信号 I = C/N です。入射量が増えても期待信号は変わりません。")
    with st.expander("画像の定義・表示処理"):
        st.latex(r"I=C/N,\qquad J=S+g(I-S),\qquad g\in\{1,2,3,5\}")
        st.caption("Sはノイズなしの理想信号、Iは標準Poisson画像、Jは表示用画像です。gはノイズ残差I−Sの振幅だけに適用し、生のIや物理計算値は変更しません。g>1のJは実撮影を再現したPoisson計数画像ではありません。")
        st.caption("計数Cの絶対SDは増加し得ますが、相対ノイズは減少します。表示した相対ノイズ・SNRは信号1の理論値です。")
        st.caption("表示窓は全画像で固定：信号 0–1.2。表示範囲外は白黒にクリップします。画面の拡大縮小はMTF測定ではありません。")
        st.caption("同じ条件・seed・パターン版・NumPy版では同じ画像を再現します。「ノイズを再生成」は条件を変えずに別の乱数標本を表示します。")
        st.caption(f"元画像は{pattern_size}×{pattern_size}画素。中央2倍は{pattern_size//2}×{pattern_size//2}領域、中央4倍は{pattern_size//4}×{pattern_size//4}領域を最近傍で表示します。拡大による情報追加や空間分解能の改善はありません。クリップ率は表示対象領域内で窓を外れた画素の割合です。")
    with st.expander("ノイズなしの理想信号"):
        ideal = ideal_signal
        st.image(prepare_display(ideal, ideal, zoom=display_zoom).uri, width=512,
                 caption="合成パターン・ノイズなし。比較画像と同じ中央領域・固定窓。")
    if result.mean_quanta < 20:
        st.warning("仮想計数が少ない領域です。画像の表示クリッピングが目立つため、見た目だけでノイズ量を評価せず理論値も確認してください。")
    with st.expander("撮影配置（犬の模式図）"):
        st.image(ROOT / "assets" / "exposure_geometry.svg", width="stretch",
                 alt="X線源から一様な被写体を透過して検出器へ進む概念図。犬のシルエットを添え、体厚tとSIDを表示。")
        st.caption("ポメラニアンは対象を示す目印です。体厚は一様物質の経路長として扱います。実寸・解剖構造・撮影姿勢・光子数を再現した図ではありません。")

with curve_tab:
    st.subheader("撮影条件に対するモデルの応答")
    axes = {"mAs": ("mas", 0.1, 20.0, "mAs"), "kVp": ("kvp", 40.0, 120.0, "kVp"),
            "SID": ("sid_cm", 50.0, 200.0, "SID (cm)"), "体厚": ("thickness_cm", 0.0, 35.0, "体厚 (cm)")}
    choice = st.selectbox("横軸にする条件", list(axes), key="curve_axis")
    field, low, high, axis_label = axes[choice]
    xs = sorted(set(np.linspace(low, high, 101).tolist() + [getattr(settings, field)]))
    curve_rows = []
    for x in xs:
        point = calculate_exposure(replace(settings, **{field: float(x)}), parameters)
        curve_rows.append({"x": x, "K": point.detector_kerma_ugy, "EI": point.ei,
                           "DI": point.di, "noise": 100 * point.relative_noise,
                           "current": x == getattr(settings, field)})
    curve_df = pd.DataFrame(curve_rows)
    st.caption("他の撮影条件・モデル係数を現在の値に固定。● は現在の条件です。破線は設定した目標を示します。")
    chart_defs = [("K", "検出器入射空気カーマ (µGy)", None), ("EI", "EI〈校正関係を仮定〉", target),
                  ("DI", "DI", 0.0), ("noise", "相対量子ノイズ (%)", None)]
    for offset in (0, 2):
        chart_cols = st.columns(2)
        for col, (y, title, reference) in zip(chart_cols, chart_defs[offset:offset + 2]):
            base = alt.Chart(curve_df).encode(x=alt.X("x:Q", title=axis_label, scale=alt.Scale(zero=False)),
                y=alt.Y(f"{y}:Q", title=title, scale=alt.Scale(zero=y != "DI")),
                tooltip=[alt.Tooltip("x:Q", title=axis_label, format=".3g"), alt.Tooltip(f"{y}:Q", title=title, format=".4g")])
            chart = base.mark_line(color="#087E8B", strokeWidth=2.5) + base.transform_filter(alt.datum.current).mark_point(color="#D97936", filled=True, size=90)
            if reference is not None:
                chart += alt.Chart(pd.DataFrame({"reference": [reference]})).mark_rule(color="#8B99A6", strokeDash=[5, 4]).encode(y="reference:Q")
            col.altair_chart(chart.properties(height=235).configure_view(stroke=None), width="stretch")
    st.caption("これらの曲線は仮定したモデルの応答です。実機の線質依存性や患者線量の変化を検証した曲線ではありません。")

with compare_tab:
    st.subheader("記録した条件を並べて比べる")
    st.caption("最大8件。撮影条件・仮定係数・目標EI・乱数seedを一緒に保存します。記録は現在のブラウザーセッション内のみです。")
    snapshots = st.session_state.snapshots
    if not snapshots:
        st.info("まず「現在の条件を記録」を押し、条件を変えてもう一度記録してください。")
    else:
        summary_rows = []
        for record in snapshots:
            s, p, r = record["settings"], record["parameters"], record["result"]
            summary_rows.append({"ID": record["id"], "記録名": record["name"], "kVp": s["kvp"], "mAs": s["mas"],
                                 "SID (cm)": s["sid_cm"], "体厚 (cm)": s["thickness_cm"],
                                 "推定K (µGy)": r["detector_kerma_ugy"], "EI（仮定）": r["ei"],
                           "EIₜ": p["target_ei"], "DI": r["di"], "相対ノイズ (%)": 100*r["relative_noise"], "理論SNR (S=1)": r["snr"], "seed": record["seed"]})
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch",
                     column_config={"推定K (µGy)": st.column_config.NumberColumn(format="%.4g"),
                                    "EI（仮定）": st.column_config.NumberColumn(format="%.4g"),
                                    "DI": st.column_config.NumberColumn(format="%+.2f"),
                                    "相対ノイズ (%)": st.column_config.NumberColumn(format="%.4g")})
        export_records = [dict(record, view_enhancement_factor=display_gain, view_zoom=display_zoom,
                               view_window_low=0.0, view_window_high=1.2,
                               view_roi="center", view_interpolation="nearest",
                               view_pattern=test_pattern, view_pattern_version=PATTERN_SPECS[test_pattern]["version"],
                               view_image_size_px=pattern_size) for record in snapshots]
        st.download_button("比較データをCSVで保存", snapshots_csv(export_records), "vet_xray_comparison.csv", "text/csv",
                           key="download_csv", help="物理計算値は未強調。記録時の表示設定と、現在の比較表示設定（view_列）を区別して出力します。relative_noiseは比、画面表示は%です。再読込機能は未実装です。")
        st.caption(f"記録画像は現在の共通表示設定（強調{display_gain}倍・中央{display_zoom}倍）で比較します。記録時の設定は保持され、CSVのview_列には現在の比較表示設定を出力します。")
        st.caption(f"記録画像も選択中の同一パターン（{pattern_size}×{pattern_size}画素）で描画します。記録時のimage_patternと表示時のview_patternは区別されます。物理計算値は変更しません。")
        if len({tuple(sorted(s["parameters"].items())) for s in snapshots}) > 1:
            st.warning("記録間でモデル係数または目標EIが異なります。撮影条件だけの比較にはなっていません。各記録の詳細を確認してください。")
        records_by_id = {s["id"]: s for s in snapshots}
        ids = list(records_by_id)
        if "image_cases" not in st.session_state:
            st.session_state.image_cases = ids[-2:]
        selected = st.multiselect("画像を並べる記録（最大4件）", ids, max_selections=4,
                                  format_func=lambda i: f"{i:02d} · {records_by_id[i]['name']}", key="image_cases")
        if selected:
            for col, identifier in zip(st.columns(len(selected)), selected):
                rec = records_by_id[identifier]
                with col:
                    st.markdown(f"**{identifier:02d} · {rec['name']}**")
                    show_simulated_image(rec["result"]["mean_quanta"], rec["seed"],
                                         f"模擬画像 · EI {rec['result']['ei']:.4g} · DI {rec['result']['di']:+.2f}")
                    st.caption(f"理論相対ノイズ {100 * rec['result']['relative_noise']:.4g}% · seed {rec['seed']}")
                    with st.expander("保存した係数・条件"):
                        st.json(rec)
        st.caption("全画像は固定表示窓 0–1.2。同じseedでも異なる計数のPoisson標本は完全な対応ノイズではありません。")
        with st.expander("記録を整理"):
            delete_ids = st.multiselect("削除する記録", ids, format_func=lambda i: f"{i:02d} · {records_by_id[i]['name']}", key="delete_ids")
            if st.button("選択した記録を削除", key="delete_cases", disabled=not delete_ids):
                st.session_state.snapshots = [s for s in snapshots if s["id"] not in delete_ids]
                for key in ("image_cases", "delete_ids"):
                    st.session_state.pop(key, None)
                st.rerun()

with model_tab:
    st.subheader("計算式・仮定・適用限界")
    st.latex(r"K_{det}=Y_{ref}\,mAs\left(\frac{kVp}{kVp_{ref}}\right)^n\left(\frac{SID_{ref}}{SID}\right)^2 e^{-\mu(kVp)t}\,T_g")
    st.latex(r"\mu(kVp)=\mu_{ref}\left(\frac{kVp_{ref}}{kVp}\right)^p,\qquad EI_{sim}=100\,K_{det}[\mu Gy],\qquad DI=10\log_{10}\!\left(\frac{EI_{sim}}{EI_T}\right)")
    st.latex(r"N=qK_{det},\quad C_i\sim\mathrm{Poisson}(N S_i),\quad I_i=C_i/N,\quad \sigma/\bar I=1/\sqrt{N}\;(S_i=1)")
    st.markdown("**IECの定義と本モデル**：IECでは画像の関心領域の代表値Vと装置固有の逆校正関数gを用いて EI = c₀g(V)、c₀ = 100 µGy⁻¹ とします。本版は校正条件での比例関係を全入力条件に仮定し、モデルKからEIを求めます。実画像の領域解析や装置校正を行っておらず、IEC適合性を示すものではありません。")
    st.markdown("**校正条件と装置依存性**：規定の線質、受像面の均一照射、後方散乱を除く空気中測定等が必要です。管電圧を合わせるだけでは校正になりません。線質、散乱、画像領域選択、検出器のエネルギー応答等で、EIと実際の入射カーマの対応は変わります。")
    st.markdown("**kVp・体厚の近似**：出力のkVp依存と実効減弱係数のkVp依存を別々に仮定しています。体厚がある場合、Kは単なるkVpの二乗ではありません。qも線質によらず一定と仮定しており、実際の光子数換算や検出効率ではありません。")
    st.latex(r"E[C_i]=NS_i,\quad SD(C_i)=\sqrt{NS_i},\quad E[I_i]=S_i,\quad SD(I_i)=\sqrt{S_i/N},\quad SNR_i=\sqrt{NS_i}")
    st.caption("Sᵢは局所の理想信号。計数Cの絶対SDと、規格化信号IのSD・相対ノイズを区別します。複数構造を含む画像全体のSDをノイズとして測定する機能はありません。表示画像は固定窓でクリップ・8ビット化され、元の線形信号とは異なります。")
    st.markdown("**DI = 0 は画質の合格判定ではありません。** 仮定した目標EIと等しいことだけを示します。MTFは空間周波数ごとの信号伝達、ノイズは信号の統計的変動です。ノイズの低下をMTFの向上と読み替えることはできません。")
    st.table(pd.DataFrame([
        {"区分": "物理的な近似", "内容": "mAsへの比例、距離の逆二乗則、均一物質の指数減弱、Poisson計数"},
        {"区分": "定義値・理論値", "内容": "EIの比例定数c₀=100 µGy⁻¹、DIの定義式、Poissonモデルに対する平均・SD・SNR"},
        {"区分": "仮定した値", "内容": "基準出力・kVp指数・減弱係数・グリッド透過率・計数換算・目標EI・合成パターン"},
        {"区分": "実測に基づく値", "内容": "この版では使用していません。実測値を登録・検証する機能は未実装"},
        {"区分": "対象外", "内容": "患者線量、散乱・ビームハードニング、実DQE、電子ノイズ、画像処理、臨床の診断能評価"},
    ]))
    st.markdown("**確認例**：mAs 2倍 → EI 2倍・DI +3.0103・相対ノイズ 1/√2。SID 2倍 → KとEI 1/4・DI −6.0206・相対ノイズ2倍。他の条件は固定します。")
    st.markdown("**一次資料**：[IEC 62494-1:2008](https://webstore.iec.ch/en/publication/7107) · [AAPM TG-116](https://www.aapm.org/pubs/reports/RPT_116.pdf)")
    st.markdown("**未実装**：ステップウェッジROI（平均・SD・SNR・CNR）、適切なエッジからのESF・LSF・MTF、DICOM読込・実測画像解析。現在の線状パターンはMTF測定画像ではありません。")
    st.caption("将来のDICOM解析では保存画素値と表示変換後の画像を分けます。処理履歴・線形性・画素間隔が不明な画像を、検出器固有のpre-sampling MTF測定に用いることはできません。")
    for filename, label in [("physics_model.md", "計算式・物理モデル説明書"),
                            ("test_results.md", "テスト結果"), ("roadmap.md", "未実装機能・開発順序")]:
        path = ROOT / "docs" / filename
        if path.exists():
            with st.expander(label):
                content = path.read_text(encoding="utf-8")
                # Repository-relative Markdown links are not Streamlit routes.
                st.markdown(re.sub(r"\[([^\]]+)\]\(((?!https?://|#)[^)]+)\)", r"\1（ローカル資料：`\2`）", content))
                st.download_button(f"{label}を保存", content, filename, "text/markdown", key=f"doc_{filename}")
    st.info("ステップウェッジ・MTF解析は第1段階の改良完了後に取り組む項目で、今回の実装には含みません。")

st.divider()
st.caption(f"VET X-RAY LAB · アプリ v{APP_VERSION} / 計算モデル v{MODEL_VERSION} · 実測データなし · 装置固有の妥当性検証は未実施")
