from __future__ import annotations


def classify_episode(ivv: float, fx: float, residual: float, total: float,
                     near_peak: float = 0.01, dominance: float = 0.60,
                     residual_share: float = 0.50) -> str:
    """Clasifica contribuciones log acumuladas; umbrales documentados en README."""
    if total >= -near_peak: return "Sin drawdown relevante"
    scale = abs(ivv) + abs(fx) + abs(residual)
    if scale == 0: return "Sin drawdown relevante"
    if abs(residual) / scale >= residual_share: return "Movimiento predominantemente residual o atípico"
    if ivv * fx < 0: return "Efectos opuestos, donde un factor compensó al otro"
    factor = abs(ivv) + abs(fx)
    if factor and abs(ivv) / factor >= dominance: return "Caída principalmente asociada al mercado estadounidense"
    if factor and abs(fx) / factor >= dominance: return "Caída principalmente asociada al USD/BRL"
    return "Caída conjunta"


def residual_description(z: float) -> str:
    az = abs(z) if z == z else 0
    if az >= 3: return "El residuo actual es extremo frente a su historia reciente."
    if az >= 2: return "El residuo actual es inusual frente a su historia reciente."
    return "El residuo actual se encuentra dentro de su rango histórico normal."
