# Verified development interpretation

The independent development audit passed for every source, score, chronological operating policy, metric, diagnostic and contrast factor before selection. The existing latest-policy anchor remains the winner at macro F2 0.3307932537, precision 0.158659 and recall 0.459837. No new addition improves that development objective.

The best new balanced addition is recent-anchor/lambda=0.25/latest, F2 0.3297257098, precision 0.167583 and recall 0.441640. Its higher precision comes with lower recall, leaving F2 below the winner. The best new unweighted addition is recent-anchor/lambda=0.25/pooled, F2 0.3280703216, precision 0.159583 and recall 0.451811; it also trails the winner. These are observed operating trade-offs, not a causal diagnosis of model training.

Among 54 otherwise matched unweighted-minus-balanced effects, 15 are positive and 39 negative. Some additions improve weaker individual anchors, but that does not establish improvement over the strongest fixed development reference. All six logit-minus-native LightGBM F2 effects are exactly zero; that transformation alone yields no measured operating gain in these cases. Every negative effect and tie remains in the complete 1199-row contrast table. Recent versus latest anchor differences change C, training weighting and spline fraction together and cannot isolate one factor.

| contrast | rows | positive | zero | negative | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| addition minus anchor | 96 | 9 | 0 | 87 | -0.009717 | 0.002493 |
| addition minus native | 96 | 74 | 0 | 22 | -0.005108 | 0.006779 |
| addition minus transformed | 96 | 74 | 0 | 22 | -0.005108 | 0.006779 |
| fixed operating reference | 600 | 121 | 7 | 472 | -0.014010 | 0.008140 |
| interaction minus additive anchor | 27 | 11 | 0 | 16 | -0.003855 | 0.005203 |
| larger lambda minus smaller lambda | 144 | 35 | 0 | 109 | -0.006553 | 0.005625 |
| policy minus own pooled | 80 | 6 | 2 | 72 | -0.009495 | 0.000805 |
| transformed minus native | 6 | 0 | 6 | 0 | 0.000000 | 0.000000 |
| unweighted minus balanced | 54 | 15 | 0 | 39 | -0.008374 | 0.004746 |

Counts summarize distinct declared scientific factors/policies, not independent experiments. They are descriptive, unadjusted for repeated adaptive search and do not alter the frozen winner. Five-reference acceptance already fails on development because selected and latest-reference F2 are identical. Consumed-month transfer and uncertainty are required only to complete this declared finite study; they cannot rescue a failed development comparison or provide independent confirmation. The broader performance goal remains unachieved.
