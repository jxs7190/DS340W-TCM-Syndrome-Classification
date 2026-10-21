# Data sources and implementation boundaries

Checked 2026-10-20. All three papers inform the research; only the first currently supplies original record-level files that can be downloaded and used here.

## 1. Gong et al. (2023): available

*A syndrome differentiation model of TCM based on multi-label deep forest using biomedical text mining.* https://doi.org/10.3389/fgene.2023.1272016

Author-cited source: https://github.com/web333panda/TCM-Dataset

Pinned commit: `b3ebee62d3adf2ee15a93c2af85fcb85963e6146`.

- `胃部疾病数据.xlsx`: adapter uses symptoms, syndrome rows, and case number (`编号`).
- `肾衰数据(1).xls`: adapter uses `ZZMC` symptoms, `ZHMC` syndrome, `YW_ID` source article. Combine identical symptom patterns within an article; keep the whole article in one split. Titles, treatment, and outcome fields are excluded.

Public raw records differ from the paper's final standardized datasets. We do not claim exact original sample counts, synonym dictionary, or results. These are source-derived records, not independently verified unique patients. Identical symptom sets are grouped across sources as an additional conservative safeguard.

No explicit dataset redistribution license was identified in the source checkout. Download locally into ignored folders; this project publishes code and aggregate evidence, not copies of the workbooks. Public access alone is not a blanket redistribution grant.

## 2. Huang et al. (2022): original records unavailable here

*A Traditional Chinese Medicine Syndrome Classification Model Based on Cross-Feature Generation by Convolution Neural Network: Model Development and Validation.* https://doi.org/10.2196/29290

Official article: https://medinform.jmir.org/2022/4/e29290/

The paper uses dysmenorrhea data from the Sichuan TCM big data management platform and literature. No record-level download was identified on the official article during this work. Its original dataset is not used in the executed runs. Authorized symptom/label records can be imported later. The generic importer does not recreate the original structured-field Cross-FGCNN architecture.

## 3. Chen et al. (2024 journal issue): request original data

*Traditional Chinese medicine diagnostic prediction model for holistic syndrome differentiation based on deep learning.* https://doi.org/10.1016/j.imr.2023.101019

Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC10826311/

Its data availability statement points to the article and the corresponding author upon reasonable request. Complete training records were not obtained here. Summary tables cannot substitute for them. This project does not claim to reproduce TCM-BERT-CNN; the original BERT/CNN experiment needs those records and a separate implementation.

## Import authorized data

Prepare a UTF-8 CSV or Excel file:

| Column | Meaning |
|---|---|
| `case_id` | Stable case/pattern ID; repeated syndrome rows are combined |
| `source_group` | Stable patient/source identifier shared by repeated visits or related source records |
| `symptoms` | Symptom text only; exclude diagnosis, treatment, and outcomes |
| `syndromes` | Actual source label names, separated by `|` for multiple labels |

Do not split compound syndrome names automatically. For structured dysmenorrhea data, prepare and document a human-reviewed symptom serialization first, preserving field meaning and missingness. The importer does not infer that mapping. Knowledge-derived samples may require guideline/document groups rather than patient IDs. Do not invent independent IDs when stronger source grouping is required.

```powershell
.venv\Scripts\python.exe -m tcm train --dataset huang --input data/raw/huang_authorized.csv --citation "DOI and actual access reference"
.venv\Scripts\python.exe -m tcm train --dataset chen --input data/raw/chen_authorized.csv --citation "DOI and actual access reference"
```

The importer records a source citation and file hash. These runs compare generic classifiers on imported data; they do not reproduce the paper architectures. Files and fitted models stay outside Git. Train each dataset separately. Broad syndrome categories, compound disease-specific syndromes, and multiclass labels cannot be pooled without an explicit expert-reviewed mapping.

## Other professional sources

No unrelated data were added merely to increase sample size. The first paper supplies two relevant datasets for the initial experiment. Other sources need symptom-to-syndrome labels, documented provenance/access terms, and source/patient identifiers. Herb databases or disease-only data cannot directly answer this question.
