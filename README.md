# Regularization in Linear Regression

This project studies how regularization changes linear regression behavior in two settings:

1. **Simple linear regression with injected outliers**
2. **Multiple linear regression with many features, including feature selection with Lasso**

The notebook compares **Ordinary Linear Regression**, **Ridge Regression**, and **Lasso Regression**, then uses **Lasso coefficients** to select important features and retrain the models on the reduced feature set.

---

## Project goal

The main goals are:

- compare ordinary linear regression, Ridge, and Lasso
- see how outliers affect fitted lines
- see how regularization affects coefficient estimates
- use Lasso as a feature selector
- compare model performance before and after feature selection

---

## Libraries used

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Main sklearn tools used:

- `LinearRegression`
- `Ridge`
- `Lasso`
- `train_test_split`
- regression metrics:
  - explained variance
  - MAE
  - MSE
  - RMSE
  - R^2

---

## A function for showing evaluation metrics

A helper function is used to print standard regression metrics for each model:

- explained variance
- `R^2`
- mean absolute error
- mean squared error
- root mean squared error

This makes it easier to compare the three regression models under the same setup.

---

## Part 1: Simple linear regression with outliers

## Fit Ordinary, Ridge, and Lasso regression models then predicting on outliers.

The notebook first creates synthetic one-feature data with an ideal linear relationship plus noise.

Plain form of the data generation:

`y = 4 + 3x + noise`

and the ideal noise-free line is:

`y_ideal = 4 + 3x`

Then a small number of artificial outliers are added to the target values for some points with larger `x` values.

### What the outlier code is doing

- `X` has one feature only
- `y` is generated from a linear relationship with random noise
- points with `X > 1.5` are eligible for outlier injection
- exactly `5` indices are selected
- large positive values are added to those `y` values

So the threshold in that part is **not** detecting outliers statistically. It is only choosing **where to inject artificial outliers**.

---

## Plotting data and predictions

### Data without outliers

The plot below shows the original noisy one-feature data together with the ideal line:

![Original data without outliers](images/original-without-outlier.png)

### Data with outliers

The next plot shows the same dataset after adding a few large outliers:

![Original data with outliers](images/original-with-outlier.png)

### Prediction comparison with outliers

The three regression models are fitted on the outlier-corrupted data and plotted against the original relationship:

![Comparison of predictions with outliers](images/compare-predictions-with-outlier.png)

### Prediction comparison without outliers

The same models are then fitted on the clean data for comparison:

![Comparison of predictions without outliers](images/compare-predictions-without-outlier.png)

We can see that ordinary linear and ridge regression performed similarly, while Lasso outperformed both.  
Although the intercept is off for the Lasso fit line, its slope is much closer to the ideal than the other fit lines.  
All three lines were pulled up by the outliers, with Lasso dampening that effect.

### Interpretation

In simple words:

- **Ordinary Linear Regression** tries to fit all points directly, so extreme outliers pull the line upward.
- **Ridge Regression** adds an `L2` penalty and shrinks coefficients, but in the one-feature outlier case it still behaves similarly to ordinary regression.
- **Lasso Regression** adds an `L1` penalty and can be more resistant to the effect of those outliers in this example.

---

## Part 2: Multiple regression regularization and lasso feature selection

### Multiple regression regularization and lasso feature selection

Here I compare performances of the three linear regression methods and then use the Lasso result to select important features to use in another model.

Note that:

- Simple linear regression: one predictor  
  `y = b0 + b1 x1`
- Multiple linear regression: two or more predictors  
  `y = b0 + b1 x1 + b2 x2 + ... + bk xk`

### Synthetic regression data

Creating synthetic regression data:

- `n_samples=100` -> create 100 data points
- `n_features=100` -> each point has 100 input features
- `n_informative=10` -> only 10 of those 100 features actually matter
- `noise=10` -> add random noise to the target values
- `random_state=42` -> makes the result reproducible
- `coef=True` -> also return the true underlying coefficients

The data is built roughly like this, suitable for regression:

`y = b + w1*x1 + w2*x2 + ... + wk*xk + noise`

This is called a regression dataset because the target `y` is continuous, and the task is to learn the mapping from features to numeric output.

### Why `coef=True` matters

The dataset is synthetic, so the generator knows the true coefficients used to build the target.  
That means we can compare:

- learned coefficients from Linear / Ridge / Lasso
- true hidden coefficients from the data generator

This gives a direct way to check whether a model recovered the real signal.

---

## First multiple-regression comparison

The three models are trained on the full 100-feature dataset and evaluated on a held-out test set.

### Prediction vs actual plots

![Multiple regression prediction vs actual](images/multiple-regression-predictionVSactual.png)

### Additional final comparison plot

![Final comparison plot](images/last-plot-comparison.png)

The results for ordinary and ridge regression are poor.  
Explained variances are under 50%, and `R^2` is very low.  
However, the result for Lasso is stellar.

### Interpretation

This happens because:

- the dataset has many irrelevant features
- only 10 of the 100 features actually matter
- ordinary regression can fit too much noise
- Ridge helps by shrinking coefficients
- Lasso helps even more because it can shrink some coefficients all the way to zero, effectively performing feature selection

---

## Model coefficients

The notebook then compares the learned coefficients with the true ideal coefficients.

![Model coefficients](images/model-coefficients.png)

![Model coefficient residuals](images/model-coefficients-residuals.png)

We can see from the first plot how much closer the Lasso coefficients are to the ideal coefficients than for the other two models. An easier way to visualize the difference is to look at the residual errors, as in the second plot. Clearly the Lasso coefficient residuals are much closer to zero than the others.

### Plain math for the coefficient comparison

If `w_true` is the ideal coefficient vector and `w_model` is a learned coefficient vector, then the residual coefficient error is:

`residual = w_true - w_model`

Smaller residuals mean the learned coefficients are closer to the true underlying signal.

---

## Using Lasso to select most important features

### Using Lasso to select most important features

Part A: Choosing a threshold value to select features based on the Lasso coefficients

A threshold is chosen by visually inspecting the Lasso coefficient residual plot. Features with sufficiently large absolute Lasso coefficients are treated as important.

The notebook uses:

`threshold = 5`

and selects features using the rule:

`abs(lasso_coefficient) > threshold`

### What I do in below codes:

1. **Fit a Lasso model**
   - Lasso gives a coefficient for each feature.

2. **Pick a cutoff**
   - if a Lasso coefficient is big enough in absolute value, treat that feature as **important**
   - if it is very close to 0, treat it as **not important**

3. **Make a table**
   - for each feature, show:
     - the Lasso coefficient
     - the true/ideal coefficient
     - whether Lasso selected it as important (`True/False`)

4. **Show two smaller tables**
   - one with only the features Lasso selected
   - one with only the features that are truly important in the generated data

5. **Compare them**
   - see whether Lasso found the truly important features correctly

So:

Use Lasso to guess which features matter, then compare its guess with the real answer.

### What is being extracted

The code is **not extracting coefficients** for later training.  
It is extracting **feature columns from `X`**.

If `important_features` contains selected column indices, then:

`X_filtered = X[:, important_features]`

means:

- keep all rows
- keep only the selected feature columns

So Lasso coefficients are used only to decide **which variables to keep**.

---

## Using the threshold to select the most important features for use in modelling

Part B: splitting data

After selecting important feature indices, the notebook builds a reduced design matrix using only those selected columns.

The printed shape is:

`(100, 10)`

So the final filtered dataset contains:

- `100` samples
- `10` selected features

This is consistent with the synthetic setup, since the dataset originally had 10 informative features.

---

## Part C: fit and apply models to those features

The models are trained again, now using only the Lasso-selected features.

This is a second-stage comparison:

- train Linear Regression on reduced features
- train Ridge on reduced features
- train Lasso on reduced features
- compare performance again

The new results are improved for ordinary and Ridge regression, and slightly improved for Lasso, supporting the idea that Lasso regression can be very beneficial when used as a feature selector.

---

## Technical workflow of the notebook

1. install and import required libraries
2. define a helper function for regression metrics
3. generate one-feature synthetic data
4. inject a few target outliers
5. fit Linear, Ridge, and Lasso on outlier and clean versions
6. visualize fitted lines
7. generate a 100-feature synthetic regression dataset
8. split into train and test sets
9. fit Linear, Ridge, and Lasso
10. compare predictions and coefficients
11. inspect coefficient residuals
12. choose a Lasso threshold
13. mark selected features in a dataframe
14. filter `X` to selected columns only
15. retrain the models on the reduced feature set
16. compare the updated results

---

## Model usage summary

### Ordinary Linear Regression

Used as the baseline model. It estimates coefficients directly without regularization.

Plain form:

`y_hat = b0 + b1*x1 + b2*x2 + ... + bk*xk`

### Ridge Regression

Used to reduce coefficient magnitude and improve stability with many features.

Objective idea:

`minimize sum((y - y_hat)^2) + alpha * sum(wj^2)`

This is `L2` regularization.

### Lasso Regression

Used both as a predictive model and as a feature selector.

Objective idea:

`minimize sum((y - y_hat)^2) + alpha * sum(|wj|)`

This is `L1` regularization.

Because of the absolute-value penalty, Lasso can push some coefficients exactly to zero, which is why it is useful for selecting important variables.

---

## Figure analysis

### `original-with-outlier.png`

Shows the one-feature dataset after a few large outliers were added. The outliers heavily stretch the y-axis and visually separate from the main linear cloud.

### `original-without-outlier.png`

Shows the same basic linear pattern without those injected outliers. The ideal line and noisy points are much more aligned.

### `compare-predictions-with-outlier.png`

Shows how the fitted lines react when outliers are present. Ordinary and Ridge are pulled more by the outliers, while Lasso stays closer to the underlying trend.

### `compare-predictions-without-outlier.png`

Shows the same models on clean data. All models become more similar, confirming that the strong distortion seen earlier came from the outliers.

### `multiple-regression-predictionVSactual.png`

Shows prediction-vs-actual comparisons for the multiple-regression case. A tighter concentration around the diagonal indicates stronger predictive performance.

### `model-coefficients.png`

Directly compares the learned coefficients against the ideal coefficients. Lasso is visually closer to the true sparse structure.

### `model-coefficients-residuals.png`

Shows coefficient errors relative to the ideal coefficients. Residuals closer to zero indicate better recovery of the true feature weights.

### `last-plot-comparison.png`

Summarizes the final comparison after feature selection, showing the effect of retraining on the reduced feature set.

---

## Final conclusion

This notebook shows two important ideas clearly:

1. **Outliers can strongly distort ordinary linear regression**
2. **Lasso can be useful both for prediction and for feature selection**

In the simple one-feature case, Lasso is less affected by injected outliers than the other two models in this example.

In the high-dimensional multiple-regression case, Lasso performs best on the full feature set and also helps identify the important variables. After reducing the data to the selected features, the ordinary and Ridge models improve noticeably, confirming that feature selection can simplify the problem and improve downstream modeling.

Overall, the notebook demonstrates that regularization is not only about controlling coefficient size, but also about improving robustness and isolating useful signal from noisy or irrelevant variables.
