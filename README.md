# PdM-OT-Module

Modulo Python generico per Predictive Maintenance (PdM) su impianti OT industriali.
Configurabile via JSON per qualsiasi tipo di impianto energetico.

Sviluppato come modulo standalone open-source nell'ambito di un progetto
di tesi ITS su sistemi OT/IT sicuri per impianti CSP.

---

## Quick Start

```
# 1. Installa le dipendenze
pip install -r requirements.txt

# 2. Avvia l'esempio fotovoltaico
python examples/fv_example.py
```

Il programma si addestra automaticamente su dati simulati e mostra
anomalie, probabilita di guasto e ticket manutenzione in tempo reale.

Per usare il tuo impianto:
- Copia config/generic_config.json
- Modifica i parametri dei tuoi sensori
- Importa i moduli nel tuo script (vedi sezione Utilizzo)

---

## Descrizione

PdM-OT-Module permette di applicare tecniche di Machine Learning per:

- Rilevare anomalie nei dati dei sensori in tempo reale (Isolation Forest)
- Predire la probabilita di guasto per zona dell'impianto (Random Forest)
- Generare ticket di manutenzione automatici con priorita e scadenza
- Salvare log strutturati in CSV per analisi successive

Il modulo e progettato per essere indipendente dal tipo di impianto.
Cambiando il file di configurazione JSON si adatta a qualsiasi contesto OT.

---

## Struttura

```
pdm-ot-module/
├── core/
│   ├── anomaly_detector.py    Isolation Forest per anomalie sensori
│   ├── fault_predictor.py     Random Forest per probabilita guasto
│   └── alert_engine.py        Gestione alert e ticket manutenzione
├── config/
│   ├── fotovoltaico_config.json   Configurazione impianto FV
│   ├── csp_config.json            Configurazione impianto CSP
│   └── generic_config.json        Template generico
├── examples/
│   └── fv_example.py          Punto di ingresso — avvia da qui
├── requirements.txt
└── README.md
```

---

## Requisiti

- Python 3.8+
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0

## Installazione

```
git clone https://github.com/tuousername/pdm-ot-module
cd pdm-ot-module
pip install -r requirements.txt
python examples/fv_example.py
```

---

## Utilizzo

```python
from core.anomaly_detector import AnomalyDetector
from core.fault_predictor  import FaultPredictor
from core.alert_engine     import AlertEngine

# Inizializza con la configurazione del tuo impianto
detector  = AnomalyDetector("config/fotovoltaico_config.json")
predictor = FaultPredictor("config/fotovoltaico_config.json")
engine    = AlertEngine("config/fotovoltaico_config.json")

# Addestra sui dati storici
detector.addestra(X_dati_normali)
predictor.addestra(X_storico, y_etichette)

# Analizza una lettura in tempo reale
lettura = {
    "kw_prodotti": 2.1,
    "temperatura_pannello": 68.0,
    "irradianza": 450.0,
    "efficienza": 35.0,
}

anomalia   = detector.analizza(lettura, zona="Campo Solare")
predizione = predictor.predici(lettura, ora=13, giorno_anno=200)

engine.processa_anomalia(anomalia)
engine.processa_predizione(predizione)
```

---

## Esempio output

```
[ CRIT ] [13:00:00] Campo Solare — Intervento immediato richiesto
         Score anomalia: 0.912
         ! Potenza prodotta: 0.8 kW (normale: 0.0-6.5)
         ! Efficienza impianto: 13.0 % (normale: 70-100)

=======================================================
  TICKET MANUTENZIONE — CRITICO
=======================================================
  ID:           TKT-20240715130012
  Zona:         Campo Solare
  Messaggio:    Verificare Stringa Nord, Stringa Sud
                Probabile guasto — intervento entro 24h
  Prob. guasto: 91%
  Scadenza:     Entro 24h dall'emissione
=======================================================
```

---

## Configurazione JSON

Per adattare il modulo al tuo impianto crea un nuovo file JSON in config/:

```json
{
  "impianto": {
    "nome": "Il Mio Impianto",
    "tipo": "custom"
  },
  "sensori": [
    {
      "id": "sensore_1",
      "nome": "Temperatura",
      "unita": "C",
      "min_normale": 20,
      "max_normale": 80,
      "peso_anomalia": 1.0
    }
  ],
  "modello": {
    "contaminazione": 0.01,
    "soglia_allarme": 0.6,
    "n_estimators": 100
  },
  "alert": {
    "livelli": {
      "basso":   {"soglia": 0.3, "messaggio": "Monitorare"},
      "medio":   {"soglia": 0.6, "messaggio": "Intervento entro 48h"},
      "alto":    {"soglia": 0.8, "messaggio": "Intervento entro 24h"},
      "critico": {"soglia": 0.9, "messaggio": "Intervento immediato"}
    }
  }
}
```

---

## Impianti supportati

| Impianto | Config | Sensori |
|---|---|---|
| Fotovoltaico (FV) | fotovoltaico_config.json | kW, temperatura pannello, irradianza, efficienza |
| Solare Termodinamico (CSP) | csp_config.json | temp fluido, sali fusi, pressione vapore, turbina |
| Generico | generic_config.json | template personalizzabile |

---

## Algoritmi ML

| Modulo | Algoritmo | Scopo |
|---|---|---|
| AnomalyDetector | Isolation Forest | Rileva comportamenti anomali nei sensori |
| FaultPredictor | Random Forest | Predice probabilita guasto per zona |

---

## Tecnologie

- Python 3
- scikit-learn (Isolation Forest, Random Forest)
- numpy
- pandas

---

## Autore

Giovanni — Studente ITS Academy Energia e Reti, Leonforte (EN)
Percorso: Tecnologie Applicate — EQF Level 5
Azienda partner: Sielte SpA

Progetto sviluppato nell'ambito di una tesi su sistemi OT/IT sicuri
per impianti CSP con modulo di Predictive Maintenance basato su ML.
