## Reinforcement Learning for Fine-Tuning Mathematical Reasoning in Language Models

This repository explores the application of Reinforcement Learning from Verifiable Rewards (RLVR), inspired by the Tulu3 paper ("Pushing Frontiers in Open Language Model Post-Training") and DeepSeek R1's GRPO, to enhance the performance of base language models on the GSM8K math problem-solving dataset.

### Key Features:

- Base Model Focus: Improves base models without relying on pre-trained reward models.

- Few-Shot Prompting: Uses few-shot examples within the model's input to establish desired data patterns and facilitate reinforcement learning.

- Simplified Prompt Format: Avoids explicit <think> and <answer> tags, keeping completions straightforward.

- Dual Reward System:

    - Correctness: Rewards accurate answers to GSM8K problems.

    - Format Adherence: Incentivizes outputting the final answer in the specific #### The final answer is {number} format.

### Goal:

The goal is to adapt and refine base models for better mathematical reasoning and structured output through RLVR, using only GSM8K data and few-shot prompting.

### Results

This section summarizes the performance improvements achieved by RL fine-tuning the Qwen/Qwen2.5-Math-1.5B model using RLVR and GRPO on the GSM8K benchmark.

![plot](./plots/RLVR-GSM8K-Plots.png)

**Performance Summary (8-shots Evaluation): **

`Qwen/Qwen2.5-Math-1.5B`:

| Metric               | Baseline (Paper) | Baseline (Mine) | After RLVR+GRPO | Improvement |
|-----------------------|------------------|-----------------|-----------------|-------------|
| GSM8K 8-shot Accuracy| 76.8             | 70.66           | 77.33           | +6.67       |


**Key Details:**

*   The model was fine-tuned for two epochs using a reward function that incorporates:
    *   **Correctness:** 1 for correct answers, 0 for incorrect.
    *   **Formatting:** 0.5 for properly formatted answers, 0 otherwise.
*   "Baseline (Mine)" refers to the model's performance on my implementation *before* RL fine-tuning.

### Implementation Notes and Considerations

This section outlines key details regarding my implementation and potential factors influencing the reported results.

*   **Resource Constraints:** Due to a 40GB VRAM limitation, I had to restrict:
    *   **Completion Length:** The maximum generated sequence length during RL fine-tuning was capped at 300 tokens.
    *   **Few-Shot Examples:** Limited to 2 examples with a maximum of 256 tokens during RL fine-tuning.

    *  **Impact:** These constraints may affect the model's ability to solve more complex problems requiring longer explanations. Removing these limitations could lead to further improvements.

*   **Dataset Ordering Bias:** The GSM8K dataset exhibits a trend where early samples may be more challenging. Without shuffling, the reported reward increase plots may appear overly optimistic due to learning on harder examples earlier in training.
    *   **Recommendation:** Shuffle the dataset to ensure unbiased evaluation and interpretation of results.

### Future Work

- **Remove Resource Constraints:** Experiment with higher VRAM environments to allow for longer completions and more few-shot examples.

- **Shuffle Dataset:** Ensure unbiased training and evaluation by shuffling the dataset.

- **Zero-Shot Evaluation:** Perform SFT on both the base and RL fine-tuned models and evaluate in a zero-shot setting for a fair comparison.