# DBbun Synthetic Missions for the DARPA Lift Challenge

## Abstract

This repository contains high-fidelity synthetic flight missions, aircraft design metadata, and generative simulation code developed by **DBbun LLC** to support engineering teams participating in the **2026 DARPA Lift Challenge**.

The synthetic mission universe includes:

- **1,000+ unique heavy-lift VTOL aircraft designs**
- Each flown through **10 independent missions**
- Resulting in **10,000+ simulated missions** and millions of telemetry samples

All scenarios follow DARPA Lift operational constraints including payload, mass, and mission time rules.

Aircraft are **bio-inspired**, drawing from nine species with superior lifting and mobility:

**Harpy eagle, Golden eagle, Osprey, Albatross, Dragonfly, Bee, Bat, Cheetah, Tiger**

Natural traits influence thrust, load structure, stability, and energy usage.

---

## Code Purpose

darpa_lift_challenge_generator_v1.0.py      # Main universe generator (unlimited # designs, unlimited # missions each)
design_viz v1.0.py           # Histograms, scatter plots, correlation heatmaps
missions_viz v1.0.py          # Mission success/failure analytics (optional)
missions_timeseries_viz v1.0.py        # Telemetry visualization per mission & per phase

---

## Dataset Files

| File                       | Description                          | Count |
|---------------------------|--------------------------------------|-------|
| `designs.csv`             | Aircraft architecture parameters     | ~1,000 |
| `missions.csv`            | Mission-level performance summaries  | ~10,000 |
| `missions_timeseries.csv` | Second-by-second telemetry logs      | 300–900 steps each |

Joinable by `design_id` and `mission_id`.

---

## Source Code

Clone and run:

```bash
git clone https://github.com/dbbun/<REPO_NAME>.git
cd <REPO_NAME>
python src/generator/mission_generator.py

## Contact & Entity Credentials
DBbun LLC
Email: contact@dbbun.com
Website: https://dbbun.com/
CAGE: 16VU3
UEI: QY39Y38E6WG8
SAM Profile: https://search.certifications.sba.gov/profile/QY39Y38E6WG8/16VU3

## Citation
Kartoun, U. (2025). DBbun Synthetic Missions for the DARPA Lift Challenge. DBbun LLC.

##License

This project is provided exclusively for internal R&D usage to support participation in the DARPA Lift Challenge.

See LICENSE.md for full restrictions and permissions.
