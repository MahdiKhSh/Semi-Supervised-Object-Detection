# Prior-art ledger (research notes, not literature-review publication)

| Work | Key idea | Relationship to project | Reference |
|---|---|---|---|
| Soft Teacher, ICCV 2021 | EMA teacher, soft classification, box jitter | Official software baseline | https://arxiv.org/abs/2106.09018 |
| Consistent-Teacher, CVPR 2023 | assignment and pseudo-target stability | Compare limitations; do not claim consistency is new | https://github.com/Adamdad/ConsistentTeacher |
| Sparse Semi-DETR, CVPR 2024 | filtering classification and localization pseudo labels separately | Similar goal; transformer architecture differs | https://arxiv.org/abs/2404.01819 |
| STEP-DETR, ICCV 2025 | stronger teacher and refined DETR queries | Potential comparison / related work | https://openaccess.thecvf.com/content/ICCV2025/html/Shehzadi_STEP-DETR_Advancing_DETR-based_Semi-Supervised_Object_Detection_with_Super_Teacher_and_ICCV_2025_paper.html |
| Building Blocks, TMLR 2025 | rare classes, GT correction and pseudo-label selection | BDD100K and data-quality relevance | https://arxiv.org/abs/2503.18903 |
| SSRT-DETR, Sensors 2026 | class/scene adaptation and multiview quality | **Strong novelty overlap**; distinct domain adaptation protocol | https://doi.org/10.3390/s26051539 |
| SSOD survey, Sensors 2026 | state-of-field literature | Review resource, not a training baseline | https://doi.org/10.3390/s26010310 |

## Claim discipline

Do not call confidence-weighted pseudo boxes, EMA teacher, localization
uncertainty filtering, adaptive class thresholds, or multiview consistency
novel by themselves. The present prototype is engineering groundwork.
A future publishable contribution requires a defensible new decision rule
and improvements in appropriately controlled experiments.

All results in external publications use their own datasets, backbones,
splits and label fractions and **are not directly comparable** with our
BDD100K results.
