"""
generate_sample_data.py
========================
Generates a SYNTHETIC, CLEARLY-LABELED sample network-flow dataset that has the
same *shape/interface* CICIDS2017 / UNSW-NB15 style flow data would have
(numeric flow-statistics features + categorical protocol/service + a Label
column), so the rest of the pipeline (preprocessing -> ML -> DL -> fusion ->
XAI) can be built, run, and tested end-to-end.

THIS IS **NOT** REAL BENCHMARK DATA.
Do NOT report metrics trained on this file as CICIDS2017/UNSW-NB15 results.
It exists purely so the codebase is runnable before you plug in the real
dataset (see README.md -> "Using the real benchmark datasets").

It also includes a "MultiStageAPT" attack class deliberately spread across
many consecutive flows (slow recon -> exploit -> exfiltrate) so the DL
temporal branch has something genuinely sequential to learn, illustrating
why a single-flow tabular model would miss it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42


def _benign_flow(rng):
    return {
        "Flow Duration": rng.normal(500, 150),
        "Total Fwd Packets": rng.poisson(12),
        "Total Bwd Packets": rng.poisson(10),
        "Fwd Packet Length Mean": rng.normal(300, 60),
        "Bwd Packet Length Mean": rng.normal(280, 60),
        "Flow Bytes/s": rng.normal(2000, 400),
        "Flow Packets/s": rng.normal(20, 5),
        "Fwd IAT Mean": rng.normal(50, 10),
        "Bwd IAT Mean": rng.normal(50, 10),
        "Packet Length Mean": rng.normal(290, 50),
        "Packet Length Std": rng.normal(40, 10),
        "SYN Flag Count": rng.integers(0, 2),
        "ACK Flag Count": rng.integers(1, 4),
        "URG Flag Count": 0,
        "Down/Up Ratio": rng.normal(1.0, 0.2),
        "Average Packet Size": rng.normal(300, 40),
        "Subflow Fwd Bytes": rng.normal(1500, 300),
        "Init Win Bytes Fwd": rng.integers(2000, 8000),
        "Init Win Bytes Bwd": rng.integers(2000, 8000),
        "Active Mean": rng.normal(100, 20),
        "Idle Mean": rng.normal(400, 100),
        "Protocol": rng.choice(["TCP", "UDP"], p=[0.8, 0.2]),
        "Service": rng.choice(["http", "https", "dns", "ftp"], p=[0.4, 0.4, 0.15, 0.05]),
    }


def _dos_flow(rng):
    f = _benign_flow(rng)
    f.update({
        "Flow Duration": rng.normal(20, 5),
        "Total Fwd Packets": rng.poisson(400),
        "Flow Bytes/s": rng.normal(50000, 8000),
        "Flow Packets/s": rng.normal(600, 100),
        "Fwd IAT Mean": rng.normal(1, 0.5),
        "SYN Flag Count": rng.integers(3, 10),
        "Down/Up Ratio": rng.normal(0.1, 0.05),
        "Protocol": "TCP",
        "Service": rng.choice(["http", "https"]),
    })
    return f


def _portscan_flow(rng):
    f = _benign_flow(rng)
    f.update({
        "Flow Duration": rng.normal(5, 2),
        "Total Fwd Packets": rng.integers(1, 3),
        "Total Bwd Packets": 0,
        "Flow Bytes/s": rng.normal(50, 20),
        "SYN Flag Count": 1,
        "ACK Flag Count": 0,
        "Init Win Bytes Bwd": 0,
        "Protocol": "TCP",
        "Service": "other",
    })
    return f


def _multistage_step_flow(rng, stage):
    """One flow within a slow multi-stage APT sequence: recon -> exploit -> exfil."""
    f = _benign_flow(rng)
    if stage == "recon":
        f.update({"Total Fwd Packets": rng.integers(1, 3), "SYN Flag Count": 1,
                   "Flow Bytes/s": rng.normal(60, 15), "Service": "other"})
    elif stage == "exploit":
        f.update({"Total Fwd Packets": rng.integers(15, 40), "SYN Flag Count": rng.integers(1, 3),
                   "Flow Bytes/s": rng.normal(3000, 500), "Service": rng.choice(["http", "ftp"])})
    else:  # exfiltrate
        f.update({"Total Fwd Packets": rng.integers(50, 150), "Flow Bytes/s": rng.normal(20000, 3000),
                   "Down/Up Ratio": rng.normal(0.05, 0.02), "Service": "ftp"})
    return f


def generate(n_benign=3000, n_dos=600, n_portscan=500, n_multistage_sequences=60,
             multistage_len=12, out_path="data/sample/sample_network_flows.csv", seed=RNG_SEED):
    """Builds a list of "events" (a single benign/DoS/PortScan flow, or an
    entire multi-stage APT chain) and SHUFFLES the event order before
    assigning a monotonically increasing global timestamp. This interleaves
    all attack classes across the whole timeline (as real traffic would be),
    so a chronological train/val/test split still sees every class in each
    split -- unlike naively concatenating one block per class."""
    rng = np.random.default_rng(seed)

    events = (["benign"] * n_benign + ["dos"] * n_dos + ["portscan"] * n_portscan +
              ["multistage"] * n_multistage_sequences)
    rng.shuffle(events)

    rows = []
    t = 0.0
    for ev in events:
        if ev == "benign":
            f = _benign_flow(rng)
            f["Timestamp"] = t
            f["Label"] = "BENIGN"
            rows.append(f)
            t += rng.uniform(0.5, 3.0)
        elif ev == "dos":
            f = _dos_flow(rng)
            f["Timestamp"] = t
            f["Label"] = "DoS"
            rows.append(f)
            t += rng.uniform(0.1, 0.5)
        elif ev == "portscan":
            f = _portscan_flow(rng)
            f["Timestamp"] = t
            f["Label"] = "PortScan"
            rows.append(f)
            t += rng.uniform(0.05, 0.3)
        else:  # multistage: a whole slow chain of flows, stays contiguous internally
            stages = (["recon"] * (multistage_len // 3) +
                      ["exploit"] * (multistage_len // 3) +
                      ["exfiltrate"] * (multistage_len - 2 * (multistage_len // 3)))
            for stage in stages:
                f = _multistage_step_flow(rng, stage)
                f["Timestamp"] = t
                f["Label"] = "MultiStageAPT"
                rows.append(f)
                t += rng.uniform(20, 90)  # deliberately slow -> evades naive rate limits

    df = pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)

    # add a few identifier-style columns that a real dataset would have and that
    # preprocessing is expected to strip out
    df["Flow ID"] = [f"flow-{i}" for i in range(len(df))]
    df["Src IP"] = [f"10.0.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(len(df))]
    df["Dst IP"] = [f"172.16.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(len(df))]

    # inject a small amount of missing / invalid values to exercise cleaning code
    for col in ["Flow Bytes/s", "Packet Length Std"]:
        idx = rng.choice(len(df), size=max(1, len(df) // 200), replace=False)
        df.loc[idx, col] = np.nan
    idx_inf = rng.choice(len(df), size=max(1, len(df) // 300), replace=False)
    df.loc[idx_inf, "Flow Packets/s"] = np.inf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[generate_sample_data] wrote {len(df)} rows -> {out_path}")
    print(df["Label"].value_counts())
    return df


if __name__ == "__main__":
    generate()
