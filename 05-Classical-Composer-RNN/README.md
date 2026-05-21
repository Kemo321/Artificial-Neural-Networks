# Project 05: Classical Composer Classifier (Sequence RNN)

## Task Description
The objective of this project is to classify classical music pieces and predict their composer based on sequential chord progressions. All input chord sequences are normalized to either the C-major or A-minor key depending on the scale of the original piece.

## Target Classes
The model maps sequences to 5 distinct composers:
* `0`: Johann Sebastian Bach
* `1`: Ludwig van Beethoven
* `2`: Claude Debussy
* `3`: Domenico Scarlatti
* `4`: Tomás Luis de Victoria

## Technical Challenges
* **Variable Sequence Lengths:** Musical pieces inherently vary in length. Processing these efficiently in mini-batches requires robust padding and packing techniques.

## Technical Approach
* **Model:** Recurrent Neural Network based on **Bi-directional LSTM** or **GRU** architectures to efficiently capture long-term musical dependencies and avoid vanishing gradients.
* **Sequence Handling Pipeline:**
  1. **Padding:** Shorter sequences are padded to match the longest sequence in the batch using custom `collate_fn` routines in the PyTorch `DataLoader`.
  2. **Embedding:** Padded sequence tokens are passed into an `nn.Embedding` layer.
  3. **Packing:** PyTorch's `pack_padded_sequence` is used before passing data into the recurrent layers, forcing the RNN to skip padding tokens during the forward pass.
  4. **Unpacking & Classification:** The final hidden state tokens from both forward and backward passes are concatenated and fed into a `nn.Linear` layer to output class probabilities.
* **Loss Function:** `nn.CrossEntropyLoss` evaluated on sequence-to-label metrics.