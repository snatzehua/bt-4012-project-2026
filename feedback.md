# Proposal Feedback

Your proposal on bank account fraud detection under temporal drift and class imbalance is an appropriate topic. There are no obvious issues regarding the dataset, preprocessing, and methodology.

The proposal doesn't specify the features included in the dataset. Feature engineering for logistic regression, random forest or gradient-boosted models is different. You can be more careful about the choices and effort in feature engineering.

My concern is that the effects of different training sets are confounded. This may come from differences in sample size or the recency of the training data. You can choose different sample sizes K and compare the differences between training with "the most recent K samples" and training with K samples from historical data.

For the final report, please provide sufficient detail and justification for each section. For example, explain your choice of dataset, its key characteristics such as size and variables, the preprocessing and feature engineering steps, and why the selected models are suitable for the tas
