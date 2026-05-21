# Project 02: Apartment Price Classifier (Imbalanced Classification)

## Task Description
The objective is to classify apartments into three distinct price categories based on specific real estate attributes and a fixed financial budget of \$350,000:
1. **Cheap:** $\le \$100,000$ (Purchasable with personal savings)
2. **Average:** $\$100,000 - \$350,000$ (Requires a mortgage loan)
3. **Expensive:** $> \$350,000$ (Out of financial reach)

## Dataset & Challenges
* **Source:** Real estate dataset containing structural attributes and the historical `SalePrice`.
* **Primary Challenge:** The dataset features highly imbalanced classes, making standard accuracy an unreliable performance metric.

## Technical Approach
* **Model:** Deep Feedforward Neural Network implemented in **PyTorch**.
* **Data Engineering:** Features such as proximity to public transport (`TimeToBusStop`, `TimeToSubway`) were mapped, and cyclical temporal features were engineered. Missing values were handled via median imputation.
* **Handling Imbalance:** Implemented **Class Weighting** within the `CrossEntropyLoss` function based on inverse class frequencies to prevent the model from biasing towards the majority class.
* **Evaluation:** Performance was measured using the **macro-average accuracy** across all individual classes.