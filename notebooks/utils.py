import re
from typing import Any
from functools import partial

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def replace_hash_with_boxed(text: str) -> str:
    """
    Replace GSM8K-style final answers (`#### 42`) with LaTeX boxed answers.

    Args:
        text: Full GSM8K answer string containing a final line like `#### 42`.

    Returns:
        The text with the last `#### <number>` converted to `\\boxed{<number>}`.
    """
    return re.sub(r"####\s*(\d+)", r"\\boxed{\1}", text)


def extract_last_after_hashes(text: str) -> str | None:
    """
    Return substring following the last occurrence of ``####``.
    """
    matches = re.findall(r"####\s*(.+)", text)
    return matches[-1] if matches else None


def extract_last_boxed(text: str) -> str | None:
    """
    Extract the content of the last \\boxed{...} in a string.

    Handles nested braces inside the box.

    Returns:
        - The content inside the last \\boxed{...}
        - None if no boxed expression is found
    """

    marker = r"\boxed{"
    start_positions = []

    # Find all occurrences of "\boxed{"
    idx = 0
    while True:
        idx = text.find(marker, idx)
        if idx == -1:
            break
        start_positions.append(idx)
        idx += len(marker)

    if not start_positions:
        return None

    # Take the last occurrence
    start = start_positions[-1] + len(marker)

    brace_count = 1
    i = start

    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count != 0:
        # Unbalanced braces
        return None

    # Extract content excluding the final closing brace
    return text[start : i - 1].strip()


def normalize_answer(ans: str) -> str | None:
    """
    Normalize numeric answers for fair comparison.

    - Remove commas
    - Strip spaces
    - Convert to float if possible
    - Remove trailing .0
    """

    if ans is None:
        return None

    ans = ans.strip()
    ans = ans.replace(",", "")

    # Remove surrounding LaTeX if present
    ans = ans.replace("$", "")

    # Remove whitespace
    ans = re.sub(r"\s+", "", ans)

    # Try numeric normalization
    try:
        num = float(ans)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except:
        return ans


def compute_accuracy(
    responses: list[str],
    dataset,
):
    """
    Compute exact match accuracy between:
    - Model boxed outputs
    - GSM8K ground truth answers
    """

    correct = 0
    total = len(responses)

    predictions = []
    references = []

    for response, example in zip(responses, dataset):

        # Extract model prediction
        pred = extract_last_boxed(response)
        pred = normalize_answer(pred)

        # Extract ground truth
        gold = example["y_true"].strip()
        gold = normalize_answer(gold).strip()

        predictions.append(pred)
        references.append(gold)

        if pred is not None and pred == gold:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "references": references,
    }


MATH_PROMPT = """
Solve the following math word problem carefully.

Think step by step to ensure the reasoning is correct.
When you are confident in the final answer, output ONLY:

\\boxed{{FINAL_ANSWER}}

Do not include any additional text in the box.

Problem:
{question}
""".strip()


def format_gsm8k_example(
    example: dict,
    *,
    system_prompt: str = "You are an expert in Math.",
    math_prompt: str = MATH_PROMPT,
) -> dict:
    """
    Build chat-style prompt and completion fields for a GSM8K example.

    Args:
        example: GSM8K example with `question` and `answer` fields.
        system_prompt: System role content for the chat prompt.
        math_prompt: User prompt template containing `{question}`.

    Returns:
        A dict with:
        - `prompt_messages`: list of chat messages for prompting.
        - `full_messages`: prompt + assistant completion messages.
        - `completion_text`: assistant target text (boxed answer format).
        - `y_true`: normalized gold answer extracted after `####`.
    """
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": math_prompt.format(question=example["question"])},
    ]

    completion_text = example["answer"].strip()
    y_true = extract_last_after_hashes(example["answer"])

    completion_text = replace_hash_with_boxed(completion_text)

    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": completion_text},
    ]

    return {
        "prompt_messages": prompt_messages,
        "full_messages": full_messages,
        "completion_text": completion_text,
        "y_true": y_true.strip().lower() if y_true is not None else None,
    }


def tokenize_chat_examples(examples: dict, tokenizer) -> dict:
    """
    Apply the tokenizer chat template to prompt and full-message sequences.

    Args:
        examples: Batched dict with `prompt_messages` and `full_messages`.
        tokenizer: Tokenizer implementing `apply_chat_template`.

    Returns:
        A dict with `prompt_ids` and `input_ids` lists.
    """
    prompt_ids = tokenizer.apply_chat_template(
        examples["prompt_messages"], tokenize=True, add_generation_prompt=True
    )

    input_ids = tokenizer.apply_chat_template(
        examples["full_messages"],
        tokenize=True,
        add_generation_prompt=False,
    )

    return {"prompt_ids": prompt_ids, "input_ids": input_ids}


def prepare_sft_sample(example: dict, *, max_length: int | None = None) -> dict:
    """
    Create masked labels for SFT where prompt tokens are ignored in the loss.

    Args:
        example: Row containing `input_ids` and `prompt_ids`.
        max_length: Optional truncation length for input and labels.

    Returns:
        A dict with `input_ids` and `labels` (prompt tokens masked with -100).
    """
    input_ids = example["input_ids"]
    prompt_len = len(example["prompt_ids"])

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_gsm8k_sft_datasets(
    *,
    tokenizer,
    train_path: str,
    eval_path: str,
    train_samples: int | None = None,
    eval_samples: int | None = None,
    seed: int = 0,
    max_length: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load GSM8K JSONL files and build SFT-ready datasets.

    This function formats examples into chat prompts, applies the chat template,
    and creates masked labels for supervised fine-tuning.

    Args:
        tokenizer: Tokenizer with `apply_chat_template`.
        train_path: Path to GSM8K train JSONL file.
        eval_path: Path to GSM8K test JSONL file.
        train_samples: Optional limit for train samples.
        eval_samples: Optional limit for eval samples.
        seed: RNG seed for training subset selection.
        max_length: Optional truncation length for sequences.

    Returns:
        (train_dataset, eval_dataset) with `input_ids`, `prompt_ids`, `labels`, `y_true`.
    """
    train_raw = Dataset.from_json(train_path, encoding="utf-8")
    eval_raw = Dataset.from_json(eval_path, encoding="utf-8")

    if train_samples is not None:
        train_raw = train_raw.shuffle(seed=seed).select(
            range(min(train_samples, len(train_raw)))
        )

    if eval_samples is not None:
        eval_raw = eval_raw.select(range(min(eval_samples, len(eval_raw))))

    tokenize_fn = partial(tokenize_chat_examples, tokenizer=tokenizer)
    prepare_fn = partial(prepare_sft_sample, max_length=max_length)

    train_formatted = train_raw.map(format_gsm8k_example)
    eval_formatted = eval_raw.map(format_gsm8k_example)

    train_formatted = train_formatted.map(tokenize_fn, batched=True)
    eval_formatted = eval_formatted.map(tokenize_fn, batched=True)

    train_tokenized = train_formatted.map(prepare_fn)
    eval_tokenized = eval_formatted.map(prepare_fn)

    keep_columns = {
        "input_ids",
        "prompt_ids",
        "labels",
        "y_true",
    }

    train_dataset = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in keep_columns]
    )
    eval_dataset = eval_tokenized.remove_columns(
        [c for c in eval_tokenized.column_names if c not in keep_columns]
    )

    return train_dataset, eval_dataset


def build_attention_mask_from_lengths(
    lengths: list[int], max_len: int | None = None
) -> torch.Tensor:
    """
    Build a padding mask from sequence lengths.

    Args:
        lengths: List of sequence lengths in a batch.
        max_len: Optional max length. Defaults to max(lengths).

    Returns:
        A tensor of shape (batch, max_len) with 1 for tokens and 0 for padding.
    """
    max_len = max(lengths) if max_len is None else max_len
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(max_len).unsqueeze(0) < lengths_tensor.unsqueeze(1)
    return mask.long()


class SFTDataCollator:
    """
    Pad variable-length SFT batches and build attention masks.
    """
    pad_token_id: int
    label_pad_token_id: int = -100

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate variable-length examples into padded tensors.

        Args:
            examples: Each example must include `input_ids` and `labels`.

        Returns:
            A batch dict with `input_ids`, `attention_mask`, and `labels`.
        """
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in examples]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in examples]

        input_lens = [len(x) for x in input_ids]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        attention_mask = build_attention_mask_from_lengths(input_lens)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class GenerationDataCollator:
    """
    Left-pad prompt-only examples so decoder-only models can generate in batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Build a padded batch for generation.

        Args:
            examples: Rows containing `prompt_ids` and `original_index`.

        Returns:
            A dict with padded `input_ids`, `attention_mask`, and `original_indices`.
        """
        batch_inputs = {
            "input_ids": [x["prompt_ids"] for x in examples],
            "attention_mask": [[1] * len(x["prompt_ids"]) for x in examples],
        }

        padded = self.tokenizer.pad(batch_inputs, return_tensors="pt")

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "original_indices": torch.tensor(
                [x["original_index"] for x in examples], dtype=torch.long
            ),
        }


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int,
    batch_size: int,
    sort_by_length: bool = True,
) -> list[str]:
    """
    Run batched greedy generation on a dataset that already contains `prompt_ids`.

    Args:
        model: Causal LM used for generation.
        tokenizer: Matching tokenizer.
        dataset: Dataset with a `prompt_ids` column.
        max_new_tokens: Maximum number of generated tokens per example.
        batch_size: Generation batch size.
        sort_by_length: Whether to sort prompts by length before batching.

    Returns:
        A list of decoded model responses in the original example order.
    """
    model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if sort_by_length:
        sorted_indices = sorted(
            range(len(dataset)), key=lambda i: len(dataset[i]["prompt_ids"])
        )
        dataset = dataset.select(sorted_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GenerationDataCollator(tokenizer),
        pin_memory=(model.device.type == "cuda"),
    )

    responses = [None] * len(dataset)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        original_indices = batch["original_indices"].tolist()

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        generated_ids = output_ids[:, input_ids.size(1) :]
        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for original_idx, text in zip(original_indices, decoded):
            responses[original_idx] = text

    tokenizer.padding_side = old_padding_side
    return responses


def compute_gsm8k_hash_accuracy(
    responses: list[str],
    dataset: Dataset,
) -> dict[str, Any]:
    """
    Compute exact-match accuracy using the answer after the last `####`.

    Args:
        responses: Generated model outputs.
        dataset: Dataset containing the gold `y_true` answers.

    Returns:
        Accuracy summary plus normalized predictions and references.
    """
    predictions = []
    references = []
    correct = 0

    for response, example in zip(responses, dataset):
        pred = normalize_answer(extract_last_after_hashes(response))
        gold = normalize_answer(example["y_true"])

        predictions.append(pred)
        references.append(gold)

        if pred is not None and pred == gold:
            correct += 1

    total = len(responses)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "references": references,
    }


def evaluate_model(
    model,
    tokenizer,
    dataset: Dataset,
    *,
    max_new_tokens: int,
    sample_count: int,
    batch_size: int,
    sort_by_length: bool = True,
) -> tuple[Dataset, list[str], dict[str, Any]]:
    """
    Evaluate a model on a prompt-tokenized GSM8K dataset.

    Args:
        model: Causal LM used for generation.
        tokenizer: Matching tokenizer.
        dataset: Dataset containing `prompt_ids` and `y_true`.
        max_new_tokens: Maximum number of generated tokens per example.
        sample_count: Number of examples to evaluate.
        batch_size: Generation batch size.
        sort_by_length: Whether to sort prompts by length before batching.

    Returns:
        The evaluated dataset subset, decoded responses, and accuracy metrics.
    """
    subset = dataset.select(range(min(sample_count, len(dataset))))
    subset = subset.add_column("original_index", list(range(len(subset))))

    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        dataset=subset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
    )

    metrics = compute_accuracy(responses=responses, dataset=subset)
    subset = subset.remove_columns(["original_index"])
    return subset, responses, metrics
