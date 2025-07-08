# PCR Prediction Accuracy (correct / total) on local testing split

| | DUKE | ISPY1 | ISPY2 | NACT | Overall Avg | Date | Notes
|-|------|-------|-------|------|-------------|------|-------|
| Predict 0 for all (baseline) | 0.6703 | 0.7910 | 0.6488 | 0.8235 | 0.6960 | July 1 |
| Joint-task epoch ~220 | 0.6703 | 0.7463 | 0.6488 | 0.8235 | 0.6863 | July 1 | 3 false positives in ISPY1, the rest are false negative
| Joint-task epoch (end of training) | 0.6703 | 0.7612 | 0.6336 | 0.7647 | 0.6703 | July 5 | 0.4242 precision, 0.15 recall