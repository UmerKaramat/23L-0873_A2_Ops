### Task 1.3 Visual EDA Summary (Generated from Actual Dataset Audit)

- **Total Files Audited:** 973
- **Valid Images Plotted:** 973
- **Corrupt Images Excluded:** 0
- **Non-Image Files Excluded:** 0
- **Detected Classes:** AI, Real

#### Per-Class Quantitative Breakdown (Valid Images Only):

| Class | Valid Images | Width Min/Med/Max (px) | Height Min/Med/Max (px) | File Size Min/Med/Max (KB) |
| :--- | :---: | :---: | :---: | :---: |
| **AI** | 538 | 225 / 1028 / 6000 | 135 / 800 / 4001 | 11.5 / 195.1 / 5494.3 |
| **Real** | 435 | 183 / 940 / 8495 | 158 / 720 / 7091 | 11.1 / 141.6 / 9956.4 |

#### Key Visual & Structural Observations:
- Image dimension consistency across classes informs input resizing decisions for Task 2 modeling.
- Formats and color channels (e.g. RGB) are verified prior to pipeline preprocessing.
- Generated plots are saved in `reports/figures/` for the final report.
