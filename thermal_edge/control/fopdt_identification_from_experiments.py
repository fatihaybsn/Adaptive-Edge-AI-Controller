#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOPDT parametre tanımlama aracı.

Bu dosya, Adaptive Edge projesindeki FPS ve image-size deney CSV'lerinden
yaklaşık bir FOPDT modeli çıkarır.

Model:
    y(t) = y0                                  , t < theta
    y(t) = y0 + K * du * (1 - exp(-(t-theta)/tau)), t >= theta

Burada:
    y(t)  : GPU sıcaklığı [°C]
    u(t)  : normalize edilmiş iş yükü girdisi
    du    : step değişimi = u_after - u_before
    K     : proses kazancı [°C / normalize yük]
    tau   : zaman sabiti [s]
    theta : ölü zaman / gecikme [s]

Not:
Bu kod mevcut CSV'lerden "yaklaşık" FOPDT parametresi çıkarır.
Akademik olarak daha güvenilir sonuç için kontrollü step deneyleri yapmak gerekir.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise SystemExit(
        "Bu script scipy gerektirir. Kurulum: pip install scipy pandas numpy matplotlib"
    ) from exc


@dataclass
class FitResult:
    dataset: str
    csv_path: str
    step_column: str
    temp_column: str
    input_definition: str
    step_time_s: float
    u_before: float
    u_after: float
    delta_u: float
    y0_c: float
    y_final_est_c: float
    K_c_per_u: float
    tau_s: float
    theta_s: float
    rmse_c: float
    r2: float
    n_points: int
    note: str


def parse_time_seconds(series: pd.Series) -> np.ndarray:
    """CSV timestamp kolonunu deney başlangıcına göre saniyeye çevirir."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.notna().all():
        return (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    # Timestamp parse edilemezse örnek numarası kullanılır.
    return np.arange(len(series), dtype=float)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Basit hareketli ortalama. Gürültüyü azaltmak için kullanılır."""
    if window <= 1:
        return x.astype(float)

    s = pd.Series(x.astype(float))
    return (
        s.rolling(window=window, center=True, min_periods=max(1, window // 3))
        .mean()
        .to_numpy(dtype=float)
    )


def fopdt_model(t_rel: np.ndarray, y0: float, delta_u: float, K: float, tau: float, theta: float) -> np.ndarray:
    """FOPDT step cevabını üretir."""
    t_eff = np.maximum(0.0, t_rel - theta)
    return y0 + K * delta_u * (1.0 - np.exp(-t_eff / tau))


def resolution_area_workload(resolution: pd.Series) -> np.ndarray:
    """
    Image size için yaklaşık normalize iş yükü.

    YOLO'da giriş boyutu küçüldükçe piksel sayısı yaklaşık karesel azalır.
    Bu yüzden 640 -> 1.00, 480 -> 0.5625, 320 -> 0.25 kabul edilir.
    """
    max_res = float(resolution.max())
    return (resolution.astype(float).to_numpy() / max_res) ** 2


def identity_workload(series: pd.Series) -> np.ndarray:
    """Kolonu doğrudan iş yükü girdisi olarak kullanır."""
    return series.astype(float).to_numpy()


def linear_resolution_workload(resolution: pd.Series) -> np.ndarray:
    """Çözünürlüğü doğrusal normalize eder. 640 -> 1.0, 320 -> 0.5 gibi."""
    max_res = float(resolution.max())
    return resolution.astype(float).to_numpy() / max_res


WORKLOAD_TRANSFORMS: Dict[str, Callable[[pd.Series], np.ndarray]] = {
    "identity": identity_workload,
    "resolution_area": resolution_area_workload,
    "resolution_linear": linear_resolution_workload,
}


def detect_step_indices(values: np.ndarray, min_separation: int = 20) -> List[int]:
    """Ayrık step kolonundaki değişim noktalarını bulur."""
    raw = np.where(values[1:] != values[:-1])[0] + 1
    if len(raw) == 0:
        return []

    # Birbirine çok yakın değişimleri tek değişim gibi ele al.
    result = [int(raw[0])]
    for idx in raw[1:]:
        if int(idx) - result[-1] >= min_separation:
            result.append(int(idx))
    return result


def fit_single_step(
    df: pd.DataFrame,
    csv_path: Path,
    dataset_name: str,
    temp_col: str,
    step_col: str,
    input_col: str,
    input_mode: str,
    step_index: int,
    pre_window_s: float,
    post_window_s: float,
    smooth_window: int,
    plot_dir: Optional[Path] = None,
) -> FitResult:
    """Bir step geçişinden FOPDT parametrelerini hesaplar."""
    if temp_col not in df.columns:
        raise ValueError(f"Sıcaklık kolonu bulunamadı: {temp_col}")
    if step_col not in df.columns:
        raise ValueError(f"Step kolonu bulunamadı: {step_col}")
    if input_col not in df.columns:
        raise ValueError(f"Girdi kolonu bulunamadı: {input_col}")
    if input_mode not in WORKLOAD_TRANSFORMS:
        raise ValueError(f"Geçersiz input_mode: {input_mode}")

    t = parse_time_seconds(df["timestamp"])
    y_raw = df[temp_col].astype(float).to_numpy()
    y = moving_average(y_raw, smooth_window)
    u = WORKLOAD_TRANSFORMS[input_mode](df[input_col])

    t_step = float(t[step_index])
    mask_pre = (t >= t_step - pre_window_s) & (t < t_step)
    mask_post = (t >= t_step) & (t <= t_step + post_window_s)

    if mask_pre.sum() < 5 or mask_post.sum() < 20:
        raise ValueError("Step çevresinde yeterli veri yok.")

    # Step öncesi sıcaklık ve iş yükü.
    y0 = float(np.nanmedian(y[mask_pre]))
    u_before = float(np.nanmedian(u[mask_pre]))
    u_after = float(np.nanmedian(u[mask_post][-max(5, int(mask_post.sum() * 0.2)):]))
    delta_u = u_after - u_before

    if abs(delta_u) < 1e-9:
        raise ValueError("Girdi değişimi çok küçük; FOPDT fit yapılamaz.")

    # Fit penceresi.
    t_fit = t[mask_post] - t_step
    y_fit = y[mask_post]

    # Başlangıç tahmini.
    y_final_est = float(np.nanmedian(y_fit[-max(5, int(len(y_fit) * 0.2)):]))
    K0 = (y_final_est - y0) / delta_u

    # Gürültülü datalarda K negatif çıkmasın diye işaret bekleneni koruyoruz.
    # Isıl yük artarsa sıcaklık artar varsayımı: K >= 0
    K0 = max(0.01, float(K0))
    tau0 = max(10.0, post_window_s / 3.0)
    theta0 = 2.0

    def residual(params: np.ndarray) -> np.ndarray:
        K, tau, theta = params
        y_hat = fopdt_model(t_fit, y0=y0, delta_u=delta_u, K=K, tau=tau, theta=theta)
        return y_hat - y_fit

    # theta post penceresinin yarısını aşmasın; tau makul ama geniş tutulur.
    lower = np.array([0.0, 1.0, 0.0], dtype=float)
    upper = np.array([200.0, max(10.0, post_window_s * 10.0), max(1.0, post_window_s * 0.5)], dtype=float)

    opt = least_squares(
        residual,
        x0=np.array([K0, tau0, theta0], dtype=float),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.2,
        max_nfev=5000,
    )

    K, tau, theta = [float(v) for v in opt.x]
    y_hat = fopdt_model(t_fit, y0=y0, delta_u=delta_u, K=K, tau=tau, theta=theta)

    rmse = float(np.sqrt(np.mean((y_hat - y_fit) ** 2)))
    ss_res = float(np.sum((y_fit - y_hat) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    result = FitResult(
        dataset=dataset_name,
        csv_path=str(csv_path),
        step_column=step_col,
        temp_column=temp_col,
        input_definition=f"{input_col} / {input_mode}",
        step_time_s=t_step,
        u_before=u_before,
        u_after=u_after,
        delta_u=delta_u,
        y0_c=y0,
        y_final_est_c=float(y0 + K * delta_u),
        K_c_per_u=K,
        tau_s=tau,
        theta_s=theta,
        rmse_c=rmse,
        r2=r2,
        n_points=int(len(t_fit)),
        note=(
            "Yaklaşık FOPDT fitidir. Mevcut CSV kontrollü ve tekrarlı step deneyi "
            "olmadığı için akademik raporda 'tahmini/identification sonucu' diye verilmelidir."
        ),
    )

    if plot_dir is not None:
        make_plot(plot_dir, result, t_fit, y_fit, y_hat)

    return result


def make_plot(plot_dir: Path, result: FitResult, t_fit: np.ndarray, y_fit: np.ndarray, y_hat: np.ndarray) -> None:
    """Fit sonucunu PNG olarak kaydeder."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t_fit, y_fit, label="Ölçülen sıcaklık")
    plt.plot(t_fit, y_hat, label="FOPDT fit")
    plt.xlabel("Step sonrası zaman [s]")
    plt.ylabel("GPU sıcaklığı [°C]")
    plt.title(f"{result.dataset} | K={result.K_c_per_u:.3f}, tau={result.tau_s:.1f}s, theta={result.theta_s:.1f}s")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    safe_name = result.dataset.lower().replace(" ", "_").replace("/", "_")
    out = plot_dir / f"{safe_name}_fopdt_fit.png"
    plt.savefig(out, dpi=160)
    plt.close()


def run_dataset(
    csv_path: Path,
    dataset_name: str,
    temp_col: str,
    step_col: str,
    input_col: str,
    input_mode: str,
    pre_window_s: float,
    post_window_s: float,
    smooth_window: int,
    plot_dir: Optional[Path],
) -> List[FitResult]:
    """CSV içindeki tüm step değişimlerini fit eder."""
    df = pd.read_csv(csv_path)

    step_values = df[step_col].to_numpy()
    step_indices = detect_step_indices(step_values)

    if not step_indices:
        raise ValueError(f"{dataset_name}: {step_col} kolonunda step değişimi bulunamadı.")

    results = []
    for step_index in step_indices:
        try:
            results.append(
                fit_single_step(
                    df=df,
                    csv_path=csv_path,
                    dataset_name=dataset_name,
                    temp_col=temp_col,
                    step_col=step_col,
                    input_col=input_col,
                    input_mode=input_mode,
                    step_index=step_index,
                    pre_window_s=pre_window_s,
                    post_window_s=post_window_s,
                    smooth_window=smooth_window,
                    plot_dir=plot_dir,
                )
            )
        except Exception as exc:
            print(f"[WARN] {dataset_name} step_index={step_index} fit edilemedi: {exc}")

    return results


def default_project_paths(project_root: Path) -> List[Tuple[str, Path, str, str, str, str]]:
    """Adaptive Edge proje klasörü için hazır veri seti tanımları."""
    return [
        (
            "FPS percentage step",
            project_root / "FPS Deney Sonuçları" / "fps_deney.csv",
            "gpu_temp",
            "percentage",
            "percentage",
            "identity",
        ),
        (
            "Image size step",
            project_root / "image-size Deney Sonuçları" / "resolution_deney.csv",
            "gpu_temperature",
            "resolution",
            "resolution",
            "resolution_area",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Edge CSV'lerinden FOPDT parametreleri çıkarır.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="adaptive-Edge proje klasörü. Varsayılan: bulunduğun klasör",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("fopdt_fit_results.json"),
        help="Sonuç JSON dosyası",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("fopdt_fit_results.csv"),
        help="Sonuç CSV dosyası",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("fopdt_fit_plots"),
        help="Fit grafiklerinin kaydedileceği klasör",
    )
    parser.add_argument("--pre-window", type=float, default=300.0, help="Step öncesi kullanılacak süre [s]")
    parser.add_argument("--post-window", type=float, default=900.0, help="Step sonrası kullanılacak süre [s]")
    parser.add_argument("--smooth-window", type=int, default=15, help="Sıcaklık için hareketli ortalama pencere sayısı")
    args = parser.parse_args()

    all_results: List[FitResult] = []

    for dataset_name, csv_path, temp_col, step_col, input_col, input_mode in default_project_paths(args.project_root):
        if not csv_path.exists():
            print(f"[WARN] CSV bulunamadı, atlandı: {csv_path}")
            continue

        results = run_dataset(
            csv_path=csv_path,
            dataset_name=dataset_name,
            temp_col=temp_col,
            step_col=step_col,
            input_col=input_col,
            input_mode=input_mode,
            pre_window_s=args.pre_window,
            post_window_s=args.post_window,
            smooth_window=args.smooth_window,
            plot_dir=args.plot_dir,
        )
        all_results.extend(results)

    if not all_results:
        raise SystemExit("Hiç FOPDT sonucu üretilemedi.")

    args.output_json.write_text(
        json.dumps([asdict(r) for r in all_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in all_results]).to_csv(args.output_csv, index=False)

    print("\nFOPDT sonuçları:")
    for r in all_results:
        print(
            f"- {r.dataset}: K={r.K_c_per_u:.4f} °C/u, "
            f"tau={r.tau_s:.1f}s, theta={r.theta_s:.1f}s, "
            f"RMSE={r.rmse_c:.3f}°C, R2={r.r2:.3f}"
        )

    print(f"\nJSON: {args.output_json}")
    print(f"CSV : {args.output_csv}")
    print(f"Plot: {args.plot_dir}")


if __name__ == "__main__":
    main()
