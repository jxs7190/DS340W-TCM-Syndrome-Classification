# Data

The parent paper reports two TCM clinical datasets: kidney disease and stomach disease. It states that the kidney dataset came from the National Population Health Science Data Centre PHDA TCM Prevention and Control of Chronic Renal Failure database, while the stomach dataset came from desensitized TCM outpatient diagnosis and treatment records. The paper points to https://github.com/web333panda/TCM-Dataset as a public source.

## Course split

For DS 340W, keep the data in three separate partitions:

- Training: 70%
- Test: 20%
- Validation (unseen): 10%

The validation set must not be used for debugging, feature selection, threshold tuning, or model choice. It should be loaded only after the training/test pipeline is finalized.

## Important grouping rule

For the stomach spreadsheet, multiple rows can represent the same clinical case because one case may have multiple syndrome labels. The `编号` column is the case identifier. Split by unique `编号`, not by individual spreadsheet rows, so the same case cannot appear in more than one partition.

## Parent-paper preprocessing

The paper describes the following preprocessing before modeling:

1. Remove records without syndrome or symptom information.
2. Combine single-label records into multi-label records.
3. Standardize symptom descriptions and merge synonymous expressions.
4. Convert symptoms to TF-IDF features.

The paper reports 436 processed stomach samples and 645 processed kidney samples. The raw spreadsheets may contain a different number of rows/case groups before all preprocessing and standardization steps are reproduced.
