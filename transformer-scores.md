# Testing Dice Scores per data set on local split

| | DUKE | ISPY1 | ISPY2 | NACT | Overall Avg | Date |
|-|------|-------|-------|------|-------------|------|
| nnUNet-baseline | 0.7810 | 0.7645 | 0.8525 | 0.7575 | 0.8067 | May 22
| 64-patch | 0.7366 | 0.7333 | 0.8118 | 0.7264 | 0.7674 | June 5
| 128-patch | 0.7532 | 0.7655 | 0.8288 | 0.7417 | 0.7876 | June 12
| 128-patch-skips | 0.7640 | 0.7613 | 0.8245 | 0.7676 | 0.7895 | June 17
| 128-patch-skips-fair | 0.7132 | 0.7723 | 0.7930 | 0.7259 | 0.7610 | June 20
| 128-patch-skips-just-fair | 0.7416 | 0.7657 | 0.8020 | 0.7505 | 0.7732 | June 23
| 128-patch-skips-just-fair (refined) | 0.7032 | 0.7542 | 0.7747 | 0.7390 | 0.7470 | July 2
| transformer-joint-more-augmentation | 0.7388 | 0.7698 | 0.7900 | 0.7441 | 0.7678 | July 5