# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
#### Model is a Random Forest Classifier with hyperparameters tuned using GridSearchCV to predict whether salary is above or below $50k using data from census that consists of age, education, hours of work per week, race, sex, native-country, relationship, occupation, martial-status, education and working class.
#### Categorical features were OneHotEncoded, while numerical features were not scaled as Random Forest was used.
<br>

## Intended Use
#### Predict if salary is above or below $50k using census data.
<br>

## Training Data
#### Source of data is from [Census data](https://archive.ics.uci.edu/dataset/20/census+income). 5-folds cross-validation on 80% of raw data. F1 score was used for scoring
<br>

## Evaluation Data
#### 20% of raw data was left for testing.
<br>

## Metrics
#### Evaluation metrics used included precision, recall and fbeta since the target class is imbalanced.
#### Metrics for evaluation data: precision 0.713672, recall 0.552597, fbeta 0.622889
<br>

## Ethical Considerations
#### The raw data only contains 32,561 entries which is not large enough to properly represent entire populations. Also there's imbalances with race and native-country, so the data and model are skewed towards the majorities.
<br>

## Caveats and Recommendations
#### Data originated from 1994 and is not contemporary, particularly after inflation it is not valid and outdated. Imbalance with race and native-country indicate predictions with those majorities are more accurate than their minorities.