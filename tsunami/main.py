"""
Tsunami Signal Extraction

Note: Keep in mind this "verification code" has the pre-known expectation
that Station 1 and Station 2 are the right stations to use.

The model will not be given these stations in order, and will have to determine
which one to use themselves!
"""

import numpy as np
from obspy import Trace, UTCDateTime

# ---------------------------------------------------------------------------
# True physical constants
# ---------------------------------------------------------------------------
RHO_TRUE   = 1027.23    # kg/m³
H_TRUE     = 2955.56    # m      — true water depth at S1 and S2
C_SOUND    = 1500.0     # m/s
G          = 9.81       # m/s²

F_RES_TRUE = C_SOUND / (4.0 * H_TRUE)

# ---------------------------------------------------------------------------
# Station geometry  (all stations lie along a single propagation line)
#
#   S1  ──50 km──  S2  ──100 km──  S3  ──50 km──  S5  ──50 km──  S4  ──50 km──  S6
#   h=3000 m       h=3000 m        h=2550 m        h=2550 m       h=3750 m       h=3750 m
# ---------------------------------------------------------------------------
STATION_SEPARATION = 50000.0    # m — S1 → S2
S3_SEPARATION      = 150000.0   # m — S1 → S3
S4_SEPARATION      = 250000.0   # m — S1 → S4
S5_SEPARATION      = 200000.0   # m — S1 → S5
S6_SEPARATION      = 300000.0   # m — S1 → S6

H_S3 = 2550.0   # m — shallower zone (continental shelf transition)
H_S4 = 3750.0   # m — deeper zone (abyssal plain)

F_C_FRACTION       = 6.25
PRE_EVENT_DURATION = 300.0   # s


# ===========================================================================
# Dispersion and shoaling helpers
# ===========================================================================

def _solve_k(omega: np.ndarray, h: float, g: float = G) -> np.ndarray:
    """Solve omega^2 = g*k*tanh(k*h) for k using Newton's method."""
    k = np.abs(omega) / max(np.sqrt(g * h), 1e-10)
    k = np.maximum(k, 1e-12)
    for _ in range(30):
        th        = np.tanh(k * h)
        residual  = g * k * th - omega ** 2
        jacobian  = g * (th + k * h * (1.0 - th ** 2))
        k = np.abs(k - residual / jacobian)
    return k


def apply_dispersive_propagation(signal: np.ndarray,
                                  t:      np.ndarray,
                                  segments: list,
                                  g: float = G) -> np.ndarray:
    """
    Add dispersive waveform distortion to a signal already positioned at its
    non-dispersive arrival time.

    segments : list of (distance_m, depth_m) pairs describing the propagation
               path from S1 to the receiving station.

    The function applies only the correction term (k_dispersive - k_nondispersive),
    preserving the arrival time while distorting the waveform shape.
    """
    dt    = t[1] - t[0]
    n     = len(t)
    freqs = np.fft.rfftfreq(n, d=dt)
    omega = 2.0 * np.pi * np.abs(freqs)
    omega[0] = 1e-10

    H = np.ones(len(freqs), dtype=complex)
    for distance, depth in segments:
        k_disp = _solve_k(omega, depth, g)
        k_nond = omega / np.sqrt(g * depth)
        H     *= np.exp(-1j * (k_disp - k_nond) * distance)
    H[0] = 1.0     # preserve DC

    S = np.fft.rfft(signal)
    return np.fft.irfft(S * H, n=n)


def green_law_factor(h_source: float, h_receiver: float) -> float:
    """Surface-displacement amplitude ratio from Green's law across a depth change."""
    return (h_source / h_receiver) ** 0.25


# ===========================================================================
# 1.  SYNTHETIC DATA GENERATION  (six stations)
# ===========================================================================

def make_tsunami_pressure(t: np.ndarray,
                          rho: float = RHO_TRUE,
                          h: float   = H_TRUE,
                          t_center: float = 1800.0) -> np.ndarray:
    T_tsunami = 600.0
    sigma     = 500.0
    eta_max   = 0.5

    eta = eta_max * np.exp(-0.5 * ((t - t_center) / sigma) ** 2) \
                  * np.cos(2 * np.pi * (t - t_center) / T_tsunami)

    return rho * G * eta


def make_seismic_pressure(t: np.ndarray,
                          rho: float = RHO_TRUE,
                          h: float   = H_TRUE,
                          seed: int  = 42) -> tuple:
    np.random.seed(seed)
    dt = t[1] - t[0]
    fs = 1.0 / dt

    t_seismic = 60.0
    decay     = 300.0
    envelope  = np.where(t >= t_seismic,
                         np.exp(-(t - t_seismic) / decay), 0.0)

    noise_raw = np.random.randn(len(t))
    tr_noise  = Trace(data=noise_raw.copy())
    tr_noise.stats.sampling_rate = fs
    tr_noise.filter('highpass', freq=0.01, corners=4, zerophase=True)
    tr_noise.filter('lowpass',  freq=0.4,  corners=4, zerophase=True)
    noise = tr_noise.data / np.std(tr_noise.data)

    phases = [
        (0.5,  80.0,  0.08),
        (0.3, 120.0,  0.06),
        (0.8, 200.0,  0.04),
        (0.2,  30.0,  0.10),
    ]
    signal = 0.02 * noise
    for amp, period, freq_offset in phases:
        signal += amp * 1e-3 * envelope \
                  * np.sin(2 * np.pi * (1.0 / period + freq_offset) * t)

    a_seismic = signal
    p_seismic = rho * h * a_seismic
    return p_seismic, a_seismic


def build_six_station_traces(dt: float = 1.0,
                               duration: float = 6000.0,
                               rho: float = RHO_TRUE,
                               h: float   = H_TRUE) -> dict:
    np.random.seed(0)

    starttime    = UTCDateTime("2025-01-01T00:00:00")
    pre_duration = PRE_EVENT_DURATION

    t_pre    = np.arange(0.0, pre_duration, dt)
    p_static = rho * G * h
    p_pre    = p_static + 0.1 * np.random.randn(len(t_pre))

    tr_pre = Trace(data=p_pre)
    tr_pre.stats.sampling_rate = 1.0 / dt
    tr_pre.stats.network   = "SY"
    tr_pre.stats.station   = "S1"
    tr_pre.stats.channel   = "BDP"
    tr_pre.stats.starttime = starttime - pre_duration

    def _tr_pre(p_data, sta):
        tr = Trace(data=p_data.copy())
        tr.stats.sampling_rate = 1.0 / dt
        tr.stats.network   = "SY"
        tr.stats.station   = sta
        tr.stats.channel   = "BDP"
        tr.stats.starttime = starttime - pre_duration
        return tr

    p_pre_s2 = rho * G * h    + 0.1 * np.random.randn(len(t_pre))
    p_pre_s3 = rho * G * H_S3 + 0.1 * np.random.randn(len(t_pre))
    p_pre_s4 = rho * G * H_S4 + 0.1 * np.random.randn(len(t_pre))
    p_pre_s5 = rho * G * H_S3 + 0.1 * np.random.randn(len(t_pre))
    p_pre_s6 = rho * G * H_S4 + 0.1 * np.random.randn(len(t_pre))

    tr_pre_s2 = _tr_pre(p_pre_s2, "S2")
    tr_pre_s3 = _tr_pre(p_pre_s3, "S3")
    tr_pre_s4 = _tr_pre(p_pre_s4, "S4")
    tr_pre_s5 = _tr_pre(p_pre_s5, "S5")
    tr_pre_s6 = _tr_pre(p_pre_s6, "S6")

    t = np.arange(0.0, duration, dt)

    c_12 = np.sqrt(G * h)
    c_s3 = np.sqrt(G * H_S3)
    c_s4 = np.sqrt(G * H_S4)

    t_center_s1 = 1800.0
    t_center_s2 = t_center_s1 + STATION_SEPARATION / c_12
    t_center_s3 = t_center_s2 + (S3_SEPARATION - STATION_SEPARATION) / c_s3
    t_center_s4 = t_center_s3 + (S4_SEPARATION - S3_SEPARATION) / c_s4
    t_center_s5 = t_center_s3 + (S5_SEPARATION - S3_SEPARATION) / c_s3
    t_center_s6 = t_center_s4 + (S6_SEPARATION - S4_SEPARATION) / c_s4

    # --- Tsunami pressure (non-dispersive timing) ---
    p_ts_s1 = make_tsunami_pressure(t, rho=rho, h=h,    t_center=t_center_s1)
    p_ts_s2 = make_tsunami_pressure(t, rho=rho, h=h,    t_center=t_center_s2)
    p_ts_s3 = make_tsunami_pressure(t, rho=rho, h=H_S3, t_center=t_center_s3)
    p_ts_s4 = make_tsunami_pressure(t, rho=rho, h=H_S4, t_center=t_center_s4)
    p_ts_s5 = make_tsunami_pressure(t, rho=rho, h=H_S3, t_center=t_center_s5)
    p_ts_s6 = make_tsunami_pressure(t, rho=rho, h=H_S4, t_center=t_center_s6)

    # --- Green's law amplitude correction ---
    g_s3_s5 = green_law_factor(h, H_S3)   # ~1.040  (wave shoals into shallower water)
    g_s4_s6 = green_law_factor(h, H_S4)   # ~0.944  (wave deepens into deeper water)
    p_ts_s3 *= g_s3_s5
    p_ts_s5 *= g_s3_s5
    p_ts_s4 *= g_s4_s6
    p_ts_s6 *= g_s4_s6

    # --- Dispersive waveform correction (applied relative to S1) ---
    # Path segments: list of (distance_m, depth_m) from S1 to each station.
    # Depth transition from h → H_S3 is assumed to occur at S2; H_S3 → H_S4 at S5.
    p_ts_s2 = apply_dispersive_propagation(p_ts_s2, t, [
        (STATION_SEPARATION, h),
    ])
    p_ts_s3 = apply_dispersive_propagation(p_ts_s3, t, [
        (STATION_SEPARATION,                          h),
        (S3_SEPARATION - STATION_SEPARATION,          H_S3),
    ])
    p_ts_s5 = apply_dispersive_propagation(p_ts_s5, t, [
        (STATION_SEPARATION,                          h),
        (S5_SEPARATION - STATION_SEPARATION,          H_S3),
    ])
    p_ts_s4 = apply_dispersive_propagation(p_ts_s4, t, [
        (STATION_SEPARATION,                          h),
        (S5_SEPARATION - STATION_SEPARATION,          H_S3),
        (S4_SEPARATION - S5_SEPARATION,               H_S4),
    ])
    p_ts_s6 = apply_dispersive_propagation(p_ts_s6, t, [
        (STATION_SEPARATION,                          h),
        (S5_SEPARATION - STATION_SEPARATION,          H_S3),
        (S6_SEPARATION - S5_SEPARATION,               H_S4),
    ])

    # --- Independent seismic noise at every station ---
    # S1 and S2 share seed=42 (same seismic zone, 50 km apart).
    # S3, S4, S5, S6 each have unique seeds so co-depth pairs cannot be
    # identified by matching noise records.
    p_seis_12, a_seis_12 = make_seismic_pressure(t, rho=rho, h=h,    seed=42)
    p_seis_s3, a_seis_s3 = make_seismic_pressure(t, rho=rho, h=H_S3, seed=17)
    p_seis_s4, a_seis_s4 = make_seismic_pressure(t, rho=rho, h=H_S4, seed=99)
    p_seis_s5, a_seis_s5 = make_seismic_pressure(t, rho=rho, h=H_S3, seed=23)
    p_seis_s6, a_seis_s6 = make_seismic_pressure(t, rho=rho, h=H_S4, seed=55)

    def _tr(data, net, sta, ch):
        tr = Trace(data=data.copy())
        tr.stats.network       = net
        tr.stats.station       = sta
        tr.stats.channel       = ch
        tr.stats.sampling_rate = 1.0 / dt
        tr.stats.starttime     = starttime
        return tr

    return {
        "s1": {
            "p_synthetic" : _tr(p_ts_s1 + p_seis_12, "SY", "S1", "BDO"),
            "a_seismic"   : _tr(a_seis_12,            "SY", "S1", "BHZ"),
            "p_tsunami"   : _tr(p_ts_s1,              "SY", "S1", "BDP"),
        },
        "s2": {
            "p_synthetic" : _tr(p_ts_s2 + p_seis_12, "SY", "S2", "BDO"),
            "a_seismic"   : _tr(a_seis_12,            "SY", "S2", "BHZ"),
        },
        "s3": {
            "p_synthetic" : _tr(p_ts_s3 + p_seis_s3, "SY", "S3", "BDO"),
            "a_seismic"   : _tr(a_seis_s3,            "SY", "S3", "BHZ"),
        },
        "s4": {
            "p_synthetic" : _tr(p_ts_s4 + p_seis_s4, "SY", "S4", "BDO"),
            "a_seismic"   : _tr(a_seis_s4,            "SY", "S4", "BHZ"),
        },
        "s5": {
            "p_synthetic" : _tr(p_ts_s5 + p_seis_s5, "SY", "S5", "BDO"),
            "a_seismic"   : _tr(a_seis_s5,            "SY", "S5", "BHZ"),
        },
        "s6": {
            "p_synthetic" : _tr(p_ts_s6 + p_seis_s6, "SY", "S6", "BDO"),
            "a_seismic"   : _tr(a_seis_s6,            "SY", "S6", "BHZ"),
        },
        "pre_event_pressure_s1"  : tr_pre,
        "pre_event_pressure_s2"  : tr_pre_s2,
        "pre_event_pressure_s3"  : tr_pre_s3,
        "pre_event_pressure_s4"  : tr_pre_s4,
        "pre_event_pressure_s5"  : tr_pre_s5,
        "pre_event_pressure_s6"  : tr_pre_s6,
        "c_tsunami_true"         : c_12,
        "delta_t_true"           : STATION_SEPARATION / c_12,
    }


# ===========================================================================
# 2.  Hydrostatic calibration
# ===========================================================================

def estimate_rho_h_hydrostatic(tr_pre_event: Trace, g: float = G) -> dict:
    p_mean  = np.mean(tr_pre_event.data)
    p_sigma = np.std(tr_pre_event.data)

    return {
        "rho_h"       : p_mean  / g,
        "rho_h_sigma" : p_sigma / g,
        "p_static"    : p_mean,
    }


# ===========================================================================
# 3.  Two-station travel time
# ===========================================================================

def detect_tsunami_arrival(tr_pressure: Trace,
                            f_detect: float = 0.005,
                            smooth_window: int = 60) -> float:
    tr_f = tr_pressure.copy()
    tr_f.filter('lowpass_cheby_2', freq=f_detect, maxorder=4)

    envelope = np.abs(tr_f.data)
    kernel   = np.ones(smooth_window) / smooth_window
    envelope = np.convolve(envelope, kernel, mode='same')

    return np.argmax(envelope) * tr_pressure.stats.delta


def estimate_depth_from_travel_time(tr_s1: Trace,
                                     tr_s2: Trace,
                                     station_sep: float,
                                     g: float = G,
                                     f_detect: float = 0.005) -> dict:
    arr_s1  = detect_tsunami_arrival(tr_s1, f_detect=f_detect)
    arr_s2  = detect_tsunami_arrival(tr_s2, f_detect=f_detect)
    delta_t = arr_s2 - arr_s1

    if delta_t <= 0:
        raise ValueError(
            f"S2 arrival ({arr_s2:.1f} s) not later than S1 ({arr_s1:.1f} s). "
            "Check station order or geometry."
        )

    c_tsunami = station_sep / delta_t

    return {
        "h_estimated" : c_tsunami ** 2 / g,
        "c_tsunami"   : c_tsunami,
        "delta_t"     : delta_t,
        "arrival_s1"  : arr_s1,
        "arrival_s2"  : arr_s2,
    }


# ===========================================================================
# 3.5  Station-pair selection
# ===========================================================================

def select_co_depth_pair(pre_event_traces: dict,
                          station_separations: dict,
                          g: float = G,
                          depth_rtol: float = 0.001) -> tuple:
    """
    Choose the best station pair from pre-event pressure records.

    Steps
    -----
    1. Estimate rho*h at every station from the mean pre-event pressure.
    2. Flag all pairs whose rho*h values agree within depth_rtol (relative).
    3. Among co-depth pairs, return the one with the smallest separation.
    """
    rho_h  = {label: np.mean(tr.data) / g
              for label, tr in pre_event_traces.items()}
    labels = sorted(rho_h)

    co_depth = [
        (la, lb)
        for i, la in enumerate(labels)
        for lb in labels[i + 1:]
        if abs(rho_h[la] - rho_h[lb]) / rho_h[la] < depth_rtol
    ]

    if not co_depth:
        raise ValueError(
            f"No co-depth pair found within depth_rtol={depth_rtol:.4f}."
        )

    def _sep(pair):
        la, lb = pair
        return (station_separations.get((la, lb))
                or station_separations[(lb, la)])

    return min(co_depth, key=_sep)


# ===========================================================================
# 4.  Parameter synthesis
# ===========================================================================

def synthesize_parameters(route2: dict,
                           route5: dict,
                           c_sound: float = C_SOUND,
                           f_c_fraction: float = F_C_FRACTION) -> dict:
    rho_h = route2["rho_h"]
    h     = route5["h_estimated"]
    rho   = rho_h / h
    f_res = c_sound / (4.0 * h)

    return {
        "rho"      : rho,
        "h"        : h,
        "rho_h"    : rho_h,
        "f_res"    : f_res,
        "f_cutoff" : f_res / f_c_fraction,
        "rho_sigma": route2["rho_h_sigma"] / h,
    }


# ===========================================================================
# 5.  Extraction
# ===========================================================================

def extract_tsunami_signal(tr_pressure: Trace,
                            tr_accel:    Trace,
                            params:      dict) -> dict:
    rho      = params["rho"]
    h        = params["h"]
    f_cutoff = params["f_cutoff"]

    nyq = tr_pressure.stats.sampling_rate / 2.0
    if f_cutoff >= nyq:
        raise ValueError(f"f_cutoff {f_cutoff:.5f} Hz >= Nyquist {nyq} Hz")

    tr_p_filt = tr_pressure.copy()
    tr_a_filt = tr_accel.copy()

    tr_p_filt.filter('lowpass', freq=f_cutoff, corners=4, zerophase=True)
    tr_a_filt.filter('lowpass', freq=f_cutoff, corners=4, zerophase=True)

    p_seis_pred          = rho * h * tr_a_filt.data
    tr_extracted         = tr_p_filt.copy()
    tr_extracted.data    = tr_p_filt.data - p_seis_pred
    tr_extracted.stats.channel = "BDE"

    return {
        "p_filtered"  : tr_p_filt,
        "p_extracted" : tr_extracted,
        "params"      : params,
    }


# ===========================================================================
# 6.  Comparison table
# ===========================================================================

def print_comparison_table(data: dict,
                            route2: dict,
                            route5: dict,
                            params: dict,
                            result: dict,
                            label_a: str,
                            label_b: str) -> None:
    true  = data["s1"]["p_tsunami"].data
    extr  = result["p_extracted"].data
    corr  = np.corrcoef(true, extr)[0, 1]
    amp_ratio = np.std(extr) / (np.std(true) + 1e-10)
    amp_penalty = np.exp(-2.0 * (np.log(amp_ratio + 1e-10)) ** 2)
    corr =  float(np.power(np.abs(corr) * amp_penalty,3))
    rms   = np.sqrt(np.mean((true - extr) ** 2))

    t_delta_t = data["delta_t_true"]
    t_c_ts    = data["c_tsunami_true"]

    rows = [
        # (label, computed, true, unit, show_pct)
        ("Station pair",   f"{label_a}–{label_b}",               "S1–S2",                            "",        False),
        ("ρh",             params["rho_h"],                       RHO_TRUE * H_TRUE,                  "kg/m²",   True),
        ("h",              params["h"],                           H_TRUE,                             "m",       True),
        ("ρ",              params["rho"],                         RHO_TRUE,                           "kg/m³",   True),
        ("f_res",          params["f_res"],                       F_RES_TRUE,                         "Hz",      True),
        ("f_cutoff",       params["f_cutoff"],                    F_RES_TRUE / F_C_FRACTION,          "Hz",      True),
        ("c_tsunami",      route5["c_tsunami"],                   t_c_ts,                             "m/s",     True),
        ("Δt (S1→S2)",     route5["delta_t"],                     t_delta_t,                          "s",       True),
        ("Extraction corr",corr,                                  1.0,                                "",        False),
        ("Extraction RMS", rms,                                   0.0,                                "Pa",      False),
    ]

    W_label = 16
    W_val   = 18
    W_true  = 18
    W_unit  = 8
    W_pct   = 10
    sep = "─" * (W_label + W_val + W_true + W_unit + W_pct + 6)

    header = (f"{'Parameter':<{W_label}}  {'Computed':>{W_val}}  "
              f"{'True':>{W_true}}  {'Unit':<{W_unit}}  {'% error':>{W_pct}}")

    print()
    print("=" * len(sep))
    print("Results summary")
    print("=" * len(sep))
    print(header)
    print(sep)

    for label, computed, true_val, unit, show_pct in rows:
        if isinstance(computed, str):
            c_str = f"{computed:>{W_val}}"
            t_str = f"{true_val:>{W_true}}"
            p_str = f"{'—':>{W_pct}}"
        elif show_pct and true_val != 0:
            pct   = 100.0 * (computed - true_val) / true_val
            c_str = f"{computed:>{W_val}.4f}"
            t_str = f"{true_val:>{W_true}.4f}"
            p_str = f"{pct:>+{W_pct}.4f}%"
        else:
            c_str = f"{computed:>{W_val}.6f}"
            t_str = f"{true_val:>{W_true}.6f}"
            p_str = f"{'—':>{W_pct}}"

        print(f"{label:<{W_label}}  {c_str}  {t_str}  {unit:<{W_unit}}  {p_str}")

    print(sep)
    print()


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    print("Building six-station synthetic traces ...")
    data = build_six_station_traces()

    pre_event_traces = {
        "S1": data["pre_event_pressure_s1"],
        "S2": data["pre_event_pressure_s2"],
        "S3": data["pre_event_pressure_s3"],
        "S4": data["pre_event_pressure_s4"],
        "S5": data["pre_event_pressure_s5"],
        "S6": data["pre_event_pressure_s6"],
    }
    station_separations = {
        ("S1", "S2"): STATION_SEPARATION,
        ("S1", "S3"): S3_SEPARATION,
        ("S1", "S4"): S4_SEPARATION,
        ("S1", "S5"): S5_SEPARATION,
        ("S1", "S6"): S6_SEPARATION,
        ("S2", "S3"): S3_SEPARATION - STATION_SEPARATION,
        ("S2", "S4"): S4_SEPARATION - STATION_SEPARATION,
        ("S2", "S5"): S5_SEPARATION - STATION_SEPARATION,
        ("S2", "S6"): S6_SEPARATION - STATION_SEPARATION,
        ("S3", "S4"): S4_SEPARATION - S3_SEPARATION,
        ("S3", "S5"): S5_SEPARATION - S3_SEPARATION,
        ("S3", "S6"): S6_SEPARATION - S3_SEPARATION,
        ("S4", "S5"): S4_SEPARATION - S5_SEPARATION,
        ("S4", "S6"): S6_SEPARATION - S4_SEPARATION,
        ("S5", "S6"): S6_SEPARATION - S5_SEPARATION,
    }
    station_data = {
        "S1": data["s1"], "S2": data["s2"],
        "S3": data["s3"], "S4": data["s4"],
        "S5": data["s5"], "S6": data["s6"],
    }

    print("Selecting co-depth station pair ...")
    label_a, label_b = select_co_depth_pair(pre_event_traces, station_separations)
    print(f"  → {label_a}–{label_b} selected")

    print("Route 2 — hydrostatic calibration ...")
    route2 = estimate_rho_h_hydrostatic(pre_event_traces[label_a])

    print(f"Route 5 — two-station travel time ({label_a}, {label_b}) ...")
    route5 = estimate_depth_from_travel_time(
        tr_s1       = station_data[label_a]["p_synthetic"],
        tr_s2       = station_data[label_b]["p_synthetic"],
        station_sep = station_separations[(label_a, label_b)],
    )

    print("Synthesizing parameters ...")
    params = synthesize_parameters(route2, route5)

    print(f"Extracting tsunami signal ({label_a}) ...")
    result = extract_tsunami_signal(
        tr_pressure = station_data[label_a]["p_synthetic"],
        tr_accel    = station_data[label_a]["a_seismic"],
        params      = params,
    )

    print_comparison_table(data, route2, route5, params, result, label_a, label_b)


if __name__ == "__main__":
    main()
