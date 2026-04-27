# Algorithmic Bias in Dynamic Pricing
### Evaluating Fairness, Profit Optimisation, and Consumer Trust Using Business Analytics

> **MSc Information Technology** · University of the West of Scotland · School of Computing · April 2026

---

## 📋 Overview

This project investigates whether algorithmic dynamic pricing systems produce systematically unfair pricing outcomes across different consumer groups. Using two real-world rideshare datasets, we implement a complete four-dimension pricing framework, train Gradient Boosting Regressor models, and conduct a structured fairness audit using the Price Disparity Ratio (PDR) metric.

**Central finding:** Every pricing dimension that increases revenue above the static baseline simultaneously introduces a price disparity that meets or exceeds the fairness alert threshold of 1.10 — confirming a genuine, measurable tension between profit optimisation and consumer fairness.

---

## 👥 Team

| Name | Banner ID | Contribution |
|------|-----------|-------------|
| Bikesh Rajbhandari | B01802393 | Implementation (Python / Jupyter Notebook) |
| Dipesh Gautam | B01820961 | Testing, Bias Audit & Results  |
| Saraswoti Pandey | B01820302 |  Literature Review |
| Nishant Lamichhane | B01820813 | Introduction, Research Design & Ethical Framework |

**Supervisor:** Michael Lin &nbsp;|&nbsp; **Moderator:** Saqid Hussain &nbsp;|&nbsp; **Programme Leader:** Jamal Hwidi

---

## 📊 Datasets
The link to the datasets used are:
https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset
https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma

| Dataset | Source | Records | Description |
|---------|--------|---------|-------------|
| `dynamic_pricing.csv` | Kaggle | 1,000 | Ride-hailing: riders, drivers, location, loyalty, ratings, booking time, vehicle, duration, cost |
| `rideshare_kaggle.csv` | Kaggle | 637,976 (cleaned) | Uber/Lyft Boston: price, distance, surge multiplier, hour, weather, service tier, route |

Both datasets are publicly available under open licences and contain no personally identifiable information.

---


## ⚙️ Implementation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook dynamic_pricing_implementation.ipynb
```

All cells use `random_state=42` — results are fully reproducible.

---

## 🔬 Methodology

### The 4 Ps Pricing Cascade

Prices are computed as a sequential multiplicative cascade following Kopalle, Pauwels, Akella & Gangwar (2023):

```
Place_adj = Period_adj × location_multiplier
Period_adj = People_adj × time_multiplier
People_adj = Historical_Cost × loyalty_multiplier × (1 + 0.05 × max(DSR − 1, 0))
```

| Dimension | Mechanism | Multipliers |
|-----------|-----------|-------------|
| **People** | Loyalty tier + demand-supply surcharge | Gold ×0.85 · Silver ×0.95 · Regular ×1.00 |
| **Product** | Vehicle tier / two-class margin model | Premium vs Economy · 12-tier ladder (DS2) |
| **Period** | Time-of-booking surge | Aft ×1.00 · Morn ×1.10 · Eve ×1.20 · Night ×1.30 |
| **Place** | Geographic demand density | Urban ×1.15 · Suburban ×1.00 · Rural ×0.90 |

### Algorithmic Model

Two **Gradient Boosting Regressor** models (Scikit-learn):

- `n_estimators=200`, `learning_rate=0.10`, `max_depth=4`, `random_state=42`
- 80/20 train-test split

| Model | Dataset | MAE | RMSE | R² | Top Feature |
|-------|---------|-----|------|----|-------------|
| GBR 1 | DS1 (1,000 rides) | £57.60 | £77.49 | **0.8353** | Ride Duration (0.9012) |
| GBR 2 | DS2 (100k sample) | £1.19 | £1.79 | **0.9633** | Service Tier (0.7791) |

### Fairness Audit

The **Price Disparity Ratio (PDR)** = max group mean ÷ min group mean. Alert threshold: **PDR > 1.10**.

| Protected Proxy | PDR | Status |
|----------------|-----|--------|
| Location — GBR model | 1.049 | ✅ Within threshold |
| Location — 4P policy | 1.213 | ⚠️ FLAGGED |
| Loyalty tier | 1.155 | ⚠️ FLAGGED |
| Time of booking | **1.255** | ⚠️ FLAGGED (highest) |
| Cab type (Uber vs Lyft) | 1.104 | ⚠️ Borderline |
| Hour group | 1.008 | ✅ Within threshold |

---

## 📈 Key Results

### Revenue Optimisation

| Pricing Regime | Total Revenue | vs Baseline |
|---------------|--------------|-------------|
| Static (uniform) | £372,502.62 | — |
| People only | £384,779.85 | +3.3% |
| People + Period | £443,004.05 | +18.9% |
| **Full 4P** | **£450,301.54** | **+20.9%** |

### Critical Finding

> The GBR model itself produces a geographically neutral PDR of **1.049** (within threshold). It is the pricing policy multipliers applied on top of the model that generate the flagged disparity of **1.213**. **Bias enters through policy design, not the algorithm.**

### Consumer Trust Risk

Surge pricing in Dataset 2 generated a **+72.9% price premium** above the base fare (£16.17 base → £27.96 surged). Per Vomberg, Homburg & Sarantopoulos (2025), premiums of this magnitude represent a significant procedural fairness violation and trust erosion risk at the pre-habituation stage.

---

## 📚 Key References

- Kopalle, P.K., Pauwels, K., Akella, L.Y. & Gangwar, M. (2023). Dynamic pricing: definition, implications for managers, and future research directions. *Journal of Retailing*, 99(4), 580–593.
- Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K. & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), Article 115.
- Vomberg, A., Homburg, C. & Sarantopoulos, P. (2025). Algorithmic dynamic pricing and consumer trust. *Journal of Marketing*, 89(1), 45–68.
- Nunan, D. & Di Domenico, M.L. (2022). Rethinking the market singularity: digital technologies and the decentralisation of market power. *European Journal of Marketing*, 56(6), 1773–1800.
- Barocas, S. & Selbst, A.D. (2016). Big data's disparate impact. *California Law Review*, 104(3), 671–732.

---

## ⚖️ Ethics

This project uses only publicly available datasets with no personally identifiable information. No human participants were involved. All datasets are properly attributed. The detection and communication of algorithmic bias is itself treated as an ethically significant activity consistent with responsible AI research practice.

---

## 📄 Licence

This repository is for academic purposes. Datasets are sourced from Kaggle under their respective open licences.
