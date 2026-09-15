---
source_type: biomedical_abstract
title: Prospective Silent Evaluation of a 30-Day Readmission Risk Model Before Clinical Deployment
journal: J Health Inform Eval
publication_year: 2024
pmid: '40010008'
authors:
- Novak T
- Ellery J
- Chaudhary S
mesh_terms:
- Patient Readmission
- Risk Assessment
- Machine Learning
---

## Background

Readmission risk models are frequently reported with retrospective discrimination metrics and deployed without prospective evaluation, leaving performance drift and workflow fit untested.

## Methods

A gradient-boosted model predicting unplanned 30-day readmission was run silently against live admissions at a 900-bed hospital for nine months, generating 24,118 scored discharges without returning predictions to clinicians. Retrospective development performance was compared against prospective performance, and decision curve analysis was used to assess net benefit across plausible intervention thresholds.

## Results

Area under the receiver operating characteristic curve fell from 0.76 in retrospective development to 0.71 prospectively. The area under the precision-recall curve was 0.28 against a readmission base rate of 14.1 percent. Calibration drifted over the evaluation period, with the expected-to-observed ratio moving from 0.98 in month 1 to 1.19 by month 9, coinciding with a documented change in discharge coding practice. Decision curve analysis showed positive net benefit only for threshold probabilities between 0.15 and 0.34.

## Conclusions

Prospective performance was materially lower than retrospective development suggested, and calibration drifted within nine months. Silent evaluation surfaced both problems before any patient was affected and should precede deployment of comparable models.
