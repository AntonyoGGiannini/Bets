"""
evaluation.py
=============

Avaliação da QUALIDADE do modelo — com foco em calibração, não só acerto.

A pergunta central (seção 15): "quando o modelo diz 60%, isso acontece perto
de 60% das vezes?". Um modelo pode acertar pouco o vencedor e ainda assim ser
excelente se suas probabilidades forem bem calibradas.

Métricas implementadas:
- Brier Score (multiclasse, para o mercado 1X2);
- Log Loss;
- Curva de calibração (dados para plotar);
- Acurácia direcional (acertou o resultado mais provável?);
- Erro médio dos gols esperados.

matplotlib é importado de forma preguiçosa apenas em ``plot_calibration_curve``
para não obrigar o ambiente a tê-lo instalado no resto do pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _as_2d(y_prob: Sequence) -> np.ndarray:
    arr = np.asarray(y_prob, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def calculate_brier_score(y_true: Sequence, y_prob: Sequence) -> float:
    """Brier Score multiclasse.

    Parameters
    ----------
    y_true:
        Vetor de inteiros com a classe verdadeira (ex.: 0=A, 1=empate, 2=B)
        OU matriz one-hot.
    y_prob:
        Matriz ``(n_amostras, n_classes)`` de probabilidades previstas.

    Returns
    -------
    float
        Média do erro quadrático entre probabilidade prevista e one-hot real.
        Menor é melhor (0 = perfeito).
    """
    probs = _as_2d(y_prob)
    n_classes = probs.shape[1]

    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim == 1:
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(y_true_arr)), y_true_arr.astype(int)] = 1.0
    else:
        onehot = y_true_arr.astype(float)

    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calculate_log_loss(y_true: Sequence, y_prob: Sequence, eps: float = 1e-15) -> float:
    """Log Loss (entropia cruzada) multiclasse. Menor é melhor."""
    probs = _as_2d(y_prob)
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)

    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim == 1:
        idx = y_true_arr.astype(int)
        picked = probs[np.arange(len(idx)), idx]
    else:
        picked = np.sum(probs * y_true_arr, axis=1)
    return float(-np.mean(np.log(picked)))


def directional_accuracy(y_true: Sequence, y_prob: Sequence) -> float:
    """Fração de vezes em que a classe de maior probabilidade foi a correta."""
    probs = _as_2d(y_prob)
    pred = np.argmax(probs, axis=1)
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim > 1:
        y_true_arr = np.argmax(y_true_arr, axis=1)
    return float(np.mean(pred == y_true_arr.astype(int)))


def calibration_curve_data(
    y_true_binary: Sequence,
    y_prob: Sequence,
    n_bins: int = 10,
):
    """Dados para a curva de calibração de um evento binário.

    Agrupa as previsões em ``n_bins`` faixas de probabilidade e compara a
    probabilidade média prevista com a frequência observada em cada faixa.

    Returns
    -------
    (mean_predicted, observed_frequency, bin_counts)
    """
    prob = np.asarray(y_prob, dtype=float)
    true = np.asarray(y_true_binary, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_pred[b] = prob[mask].mean()
            obs_freq[b] = true[mask].mean()
    return mean_pred, obs_freq, counts


def plot_calibration_curve(
    y_true_binary: Sequence,
    y_prob: Sequence,
    n_bins: int = 10,
    title: str = "Curva de Calibração",
    save_path: Optional[str] = None,
):
    """Plota a curva de calibração. Requer matplotlib (import preguiçoso).

    Se ``save_path`` for informado, salva a figura; caso contrário retorna o
    objeto ``Figure`` para o chamador exibir.
    """
    try:
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib é necessário para plot_calibration_curve") from exc

    mean_pred, obs_freq, counts = calibration_curve_data(y_true_binary, y_prob, n_bins)
    valid = ~np.isnan(mean_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    ax.plot(mean_pred[valid], obs_freq[valid], "o-", color="#1f77b4", label="Modelo")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def evaluate_goal_prediction(actual_goals: Sequence, expected_goals: Sequence) -> dict:
    """Erro dos gols esperados: MAE e RMSE entre lambdas e gols reais."""
    actual = np.asarray(actual_goals, dtype=float)
    expected = np.asarray(expected_goals, dtype=float)
    err = expected - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }
