"""
╔══════════════════════════════════════════════════════════════╗
║  PdM-OT-Module — Esempio Fotovoltaico                       ║
║  Simula un impianto FV e applica anomaly detection + PdM    ║
╚══════════════════════════════════════════════════════════════╝

REQUISITI:
    pip install scikit-learn numpy pandas
"""

import sys
import os
import math
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.anomaly_detector import AnomalyDetector
from core.fault_predictor  import FaultPredictor
from core.alert_engine     import AlertEngine

# ── Percorso config ──────────────────────────────────────────
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "fotovoltaico_config.json")

# ── Dati ENEA Sicilia ────────────────────────────────────────
IRRADIANZA = {1:2.8,2:3.5,3:4.8,4:5.9,5:6.8,6:7.5,7:7.8,8:7.2,9:5.6,10:4.1,11:2.9,12:2.5}
ORE_LUCE   = {1:9,2:10,3:12,4:13,5:14,6:15,7:15,8:14,9:12,10:11,11:10,12:9}

def simula_lettura(ora: int, mese: int, guasto: bool = False) -> dict:
    """Genera una lettura realistica dei sensori FV."""
    ore  = ORE_LUCE[mese]
    alba = 12 - ore / 2
    tram = 12 + ore / 2

    if ora <= alba or ora >= tram:
        irr = 0.0
    else:
        angolo = (ora - alba) / (tram - alba) * 180
        irr = max(0.0, math.sin(math.radians(angolo)))

    irr_mensile = IRRADIANZA[mese]
    temp_pannello = 25 + irr * 30 + random.uniform(-3, 3)
    coeff_temp = 1 - 0.004 * max(0, temp_pannello - 25)
    kw = round(6.0 * irr * (irr_mensile / 7.8) * coeff_temp * random.uniform(0.95, 1.05), 3)
    irradianza_wm2 = round(irr * irr_mensile / 7.8 * 1000, 1)
    efficienza = round((kw / 6.0 * 100) if kw > 0 else 0, 1)

    if guasto and kw > 0:
        kw = round(kw * random.uniform(0.1, 0.35), 3)
        efficienza = round(kw / 6.0 * 100, 1)

    return {
        "kw_prodotti":        kw,
        "temperatura_pannello": round(temp_pannello, 1),
        "irradianza":          irradianza_wm2,
        "efficienza":          efficienza,
    }

def genera_dataset_training(n_giorni: int = 180):
    """Genera dataset di training (dati normali + guasti etichettati)."""
    X, y = [], []
    for giorno in range(n_giorni):
        mese = (giorno // 30 % 12) + 1
        for ora in range(24):
            guasto = random.random() < 0.02
            l = simula_lettura(ora, mese, guasto=guasto)
            features = list(l.values()) + [ora / 23.0, mese / 12.0]
            X.append(features)
            y.append(1 if guasto else 0)
    return np.array(X), np.array(y)

def main():
    print("\n" + "="*55)
    print("  PdM-OT-Module — Esempio Fotovoltaico")
    print("="*55)

    # Inizializza moduli
    detector   = AnomalyDetector(CONFIG)
    predictor  = FaultPredictor(CONFIG)
    engine     = AlertEngine(CONFIG, log_dir="logs/fv")

    # Genera dataset e addestra
    print("\n[1/3] Generazione dataset training (180 giorni)...")
    X, y = genera_dataset_training(180)
    X_anomaly = X[:, :4]   # solo features sensori per anomaly detector

    print("[2/3] Addestramento modelli...")
    detector.addestra(X_anomaly)
    predictor.addestra(X, y)

    # Simulazione real-time — 24 ore di Luglio
    print("\n[3/3] Simulazione real-time — Luglio (mese 7)")
    print("─"*55)

    for ora in range(24):
        # Simula guasto alle 13
        guasto = (ora == 13)
        lettura = simula_lettura(ora, 7, guasto=guasto)

        # Anomaly detection
        anomalia = detector.analizza(lettura, zona="Campo Solare")
        engine.processa_anomalia(anomalia, stampa=(anomalia["livello"] != "ok"))

        # Fault prediction
        predizione = predictor.predici(lettura, ora=ora, giorno_anno=200)
        engine.processa_predizione(predizione, stampa=(predizione["prob_guasto"] > 0.5))

        # Stampa lettura
        kw  = lettura["kw_prodotti"]
        eff = lettura["efficienza"]
        tag = " << GUASTO SIMULATO" if guasto else ""
        print(f"  Ora {ora:02d}: {kw:.2f} kW  |  efficienza {eff:.0f}%{tag}")

    # Summary finale
    engine.summary()
    print(f"\n  Metriche modello: {predictor.metriche_modello()}")

if __name__ == "__main__":
    main()
