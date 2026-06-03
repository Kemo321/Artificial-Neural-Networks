# Project 06: Hate Speech Classification (NLP)

## Task Description
The objective of this project is to build a robust classification model capable of detecting hate speech in Polish-language comments. The model processes tagged text data to distinguish between offensive/hate speech and neutral content.

## Technical Approach
* **Model Architecture:** The project utilizes **HerBERT** (`allegro/herbert-base-cased`), a Transformer-based language model specifically pre-trained on Polish corpora.
* **Learning Strategy:** Fine-tuning. The pre-trained weights are adjusted to the specific target labels provided in the `hate_train.csv` dataset, leveraging transfer learning to achieve high accuracy with limited training samples.
* **Pipeline:**
    1.  **Preprocessing:** Tokenization of raw text using the `HerbertTokenizer`, handling truncation and padding to ensure consistent input tensor shapes.
    2.  **Fine-tuning:** Training the sequence classification head on top of the transformer base, utilizing Cross-Entropy loss.
    3.  **Inference:** Applying the trained model to `hate_test_data.txt` to generate class predictions for the unlabeled test set.

## Technical Pipeline
* **Tokenization:** Converting text to input IDs and attention masks optimized for the HerBERT model.
* **Classification Head:** A linear layer added on top of the transformer's pooler output, mapping the feature vector to the target classes (hate vs. non-hate).
* **Optimization:** Stochastic Gradient Descent or AdamW optimizer used for weight updates during the fine-tuning process.

## How to run
1. Ensure the training data (`hate_train.csv`) and test data (`hate_test_data.txt`) are in the project root.
2. Install dependencies: `pip install torch transformers pandas scikit-learn`
3. Run `solution.ipynb` using Jupyter Notebook or VS Code.
4. The script will automatically generate the `pred.csv` file containing the predictions as required.

## Project Structure
- `solution.ipynb`: Main notebook for data processing, model training, and inference.
- `pred.csv`: Output file containing class predictions (single column, no header).
- `hate_train.csv`: Training dataset.
- `hate_test_data.txt`: Unlabeled test dataset.