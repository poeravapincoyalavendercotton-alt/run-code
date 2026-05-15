"""Tsunami Extraction oracle
"""
import json
import numpy as np
from obspy import Trace, UTCDateTime

# --- Hidden parameters (model never sees this) ---
RHO   = 1027.23    # kg/m³
H     = 2955.56    # m      — true water depth at S1 and S2
P_TSUNAMI = 0   # replaced when traces are generated
C_TSUNAMI = 170.2764
F_RES     = 0.1269
F_CUTOFF  = 0.02

# ---------------------------------------------------------------------------
# Station geometry  (all stations lie along a single propagation line)
#
#   S1  ──50 km──  S2  ──100 km──  S3  ──50 km──  S5  ──50 km──  S4  ──50 km──  S6
#   h=3000 m       h=3000 m        h=2550 m        h=2550 m       h=3750 m       h=3750 m
# ---------------------------------------------------------------------------
STATION_SEPARATION = 50000.0
S3_SEPARATION      = 150000.0
S4_SEPARATION      = 250000.0
S5_SEPARATION      = 200000.0
S6_SEPARATION      = 300000.0

H_S3 = 2550.0
H_S4 = 3750.0

F_C_FRACTION       = 6.25
PRE_EVENT_DURATION = 300.0

G       = 9.81
C_SOUND = 1500


# ===========================================================================
# Dispersion and shoaling helpers
# ===========================================================================

def _solve_k(omega: np.ndarray, h: float, g: float = G) -> np.ndarray:
    """Solve omega^2 = g*k*tanh(k*h) for k using Newton's method."""
    k = np.abs(omega) / max(np.sqrt(g * h), 1e-10)
    k = np.maximum(k, 1e-12)
    for _ in range(30):
        th       = np.tanh(k * h)
        residual = g * k * th - omega ** 2
        jacobian = g * (th + k * h * (1.0 - th ** 2))
        k = np.abs(k - residual / jacobian)
    return k


def apply_dispersive_propagation(signal: np.ndarray,
                                  t:      np.ndarray,
                                  segments: list,
                                  g: float = G) -> np.ndarray:
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
    H[0] = 1.0

    S = np.fft.rfft(signal)
    return np.fft.irfft(S * H, n=n)


def green_law_factor(h_source: float, h_receiver: float) -> float:
    return (h_source / h_receiver) ** 0.25


# ===========================================================================
# Synthetic data generation
# ===========================================================================

def make_tsunami_pressure(t: np.ndarray,
                          rho: float = RHO,
                          h: float   = H,
                          t_center: float = 1800.0) -> np.ndarray:
    T_tsunami = 600.0
    sigma     = 500.0
    eta_max   = 0.5

    eta = eta_max * np.exp(-0.5 * ((t - t_center) / sigma) ** 2) \
                  * np.cos(2 * np.pi * (t - t_center) / T_tsunami)

    return rho * G * eta


def make_seismic_pressure(t: np.ndarray,
                          rho: float = RHO,
                          h: float   = H,
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


def trace_to_dict(trace):
    stats = dict(trace.stats)
    stats["starttime"] = str(stats["starttime"])
    stats["endtime"]   = str(stats["endtime"])
    return {"data": trace.data.tolist(), "stats": stats}


def build_six_station_traces(dt: float = 1.0,
                               duration: float = 6000.0,
                               rho: float = RHO,
                               h: float   = H) -> dict:
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

    # Tsunami pressure (non-dispersive timing)
    p_ts_s1 = make_tsunami_pressure(t, rho=rho, h=h,    t_center=t_center_s1)
    p_ts_s2 = make_tsunami_pressure(t, rho=rho, h=h,    t_center=t_center_s2)
    p_ts_s3 = make_tsunami_pressure(t, rho=rho, h=H_S3, t_center=t_center_s3)
    p_ts_s4 = make_tsunami_pressure(t, rho=rho, h=H_S4, t_center=t_center_s4)
    p_ts_s5 = make_tsunami_pressure(t, rho=rho, h=H_S3, t_center=t_center_s5)
    p_ts_s6 = make_tsunami_pressure(t, rho=rho, h=H_S4, t_center=t_center_s6)

    # Green's law amplitude correction
    g_s3_s5 = green_law_factor(h, H_S3)
    g_s4_s6 = green_law_factor(h, H_S4)
    p_ts_s3 *= g_s3_s5
    p_ts_s5 *= g_s3_s5
    p_ts_s4 *= g_s4_s6
    p_ts_s6 *= g_s4_s6

    # Dispersive waveform correction (relative to S1)
    p_ts_s2 = apply_dispersive_propagation(p_ts_s2, t, [
        (STATION_SEPARATION, h),
    ])
    p_ts_s3 = apply_dispersive_propagation(p_ts_s3, t, [
        (STATION_SEPARATION,                 h),
        (S3_SEPARATION - STATION_SEPARATION, H_S3),
    ])
    p_ts_s5 = apply_dispersive_propagation(p_ts_s5, t, [
        (STATION_SEPARATION,                 h),
        (S5_SEPARATION - STATION_SEPARATION, H_S3),
    ])
    p_ts_s4 = apply_dispersive_propagation(p_ts_s4, t, [
        (STATION_SEPARATION,                 h),
        (S5_SEPARATION - STATION_SEPARATION, H_S3),
        (S4_SEPARATION - S5_SEPARATION,      H_S4),
    ])
    p_ts_s6 = apply_dispersive_propagation(p_ts_s6, t, [
        (STATION_SEPARATION,                 h),
        (S5_SEPARATION - STATION_SEPARATION, H_S3),
        (S6_SEPARATION - S5_SEPARATION,      H_S4),
    ])

    # Independent seismic noise — unique seed per station
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

    global P_TSUNAMI
    P_TSUNAMI = _tr(p_ts_s1, "SY", "S1", "BDP").data

    return {
        "s1": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s1 + p_seis_12, "SY", "S1", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_12,            "SY", "S1", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre)['data']
        },
        "s2": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s2 + p_seis_12, "SY", "S2", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_12,            "SY", "S2", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre_s2)['data']
        },
        "s3": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s3 + p_seis_s3, "SY", "S3", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_s3,            "SY", "S3", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre_s3)['data']
        },
        "s4": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s4 + p_seis_s4, "SY", "S4", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_s4,            "SY", "S4", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre_s4)['data']
        },
        "s5": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s5 + p_seis_s5, "SY", "S5", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_s5,            "SY", "S5", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre_s5)['data']
        },
        "s6": {
            "p_synthetic" : trace_to_dict(_tr(p_ts_s6 + p_seis_s6, "SY", "S6", "BDO"))['data'],
            "a_seismic"   : trace_to_dict(_tr(a_seis_s6,            "SY", "S6", "BHZ"))['data'],
            "pre_event_pressure"  : trace_to_dict(tr_pre_s6)['data']
        }
    }


def generate_similarity(parameters: dict) -> float:
    if 'p_tsunami' not in parameters:
        raise ValueError("Missing 'p_tsunami' parameter.")
    p = parameters['p_tsunami']
    if isinstance(p, str):
        ext = p.rsplit('.', 1)[-1].lower()
        if ext == 'npy':
            p = np.load(p)
        elif ext == 'json':
            with open(p) as f:
                p = np.array(json.load(f))
        else:
            p = np.loadtxt(p)
    p = np.asarray(p, dtype=float)
    corr = np.corrcoef(P_TSUNAMI, p)[0, 1]
    amp_ratio = np.std(p) / (np.std(P_TSUNAMI) + 1e-10)
    amp_penalty = np.exp(-2.0 * (np.log(amp_ratio + 1e-10)) ** 2)
    return np.abs(corr) * amp_penalty


TRACES = build_six_station_traces()
SIMILARITY_CALLED = False
COMPARISON_VALUE = None


def handle_query(mode: str, parameters: dict) -> dict:
    """Called when the model invokes query_oracle."""

    if mode == "help":
        return {
            "type": "tsunami_wave_extraction",
            "description": (
                "Synthetic combined pressure records, vertical accelerations, "
                "and pre-event pressure records from 6 stations."
            ),
            "modes": {
                "measurements_1": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "measurements_2": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "measurements_3": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "measurements_4": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "measurements_5": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "measurements_6": {
                    "description": (
                        "Returns the tsunami + seismic combined pressure records, "
                        "vertical accelerations, and pre-event pressure records from a station."
                    ),
                    "parameters": {}
                },
                "tsunami_comparison": {
                    "description": (
                        "Returns a numerical comparison from the submitted tsunami wave"
                    ),
                    "parameters": {
                        "p_tsunami": (
                            "An array consisting of the extracted tsunami wave after "
                            "seismic pressure prediction. Length 6000, all floats. "
                            "May also be a path to a JSON file."
                        )
                    },
                    "returns": {
                        "comparison": (
                            "Comparison value between the extracted tsunami wave "
                            "and the true tsunami wave."
                        )
                    }
                }
            }
        }

    elif mode == "tsunami_comparison":
        global SIMILARITY_CALLED, COMPARISON_VALUE
        if not SIMILARITY_CALLED:
            raw = np.abs(generate_similarity(parameters))
            SIMILARITY_CALLED = True
            COMPARISON_VALUE = float(np.power(raw,3))
            return float(np.power(raw,3))
        return COMPARISON_VALUE

    elif mode == 'measurements_1':
        return TRACES['s3']
    elif mode == 'measurements_2':
        return TRACES['s4']
    elif mode == 'measurements_3':
        return TRACES['s1']
    elif mode == 'measurements_4':
        return TRACES['s5']
    elif mode == 'measurements_5':
        return TRACES['s2']
    elif mode == 'measurements_6':
        return TRACES['s6']
