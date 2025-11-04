#!/usr/bin/env python3
"""Train a mushroom classifier and generate submission predictions.

This script implements a multinomial Naive Bayes classifier with feature
engineering tailored for the mushroom dataset. It performs stratified
cross-validation to estimate macro F1-score, trains on the full dataset,
and writes predictions for the provided test set to ``submission.csv``.
"""

import csv
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_COLUMN = "kelas"
ID_COLUMN = "id"
TEXT_COLUMN = "deskripsi_singkat"
RANDOM_SEED = 42


def load_dataset(path: str, target_column: Optional[str] = None) -> Tuple[List[Dict[str, str]], Optional[List[str]], Optional[List[str]]]:
    """Load a CSV dataset.

    Returns a tuple of (rows, feature_columns, labels). For inference datasets
    that do not include the target column, the second and third return values
    are ``None``.
    """

    with open(path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: List[Dict[str, str]] = []
        labels: List[str] = []
        for raw_row in reader:
            row = {key: (value.strip() if value is not None else "") for key, value in raw_row.items()}
            if target_column and target_column in row:
                labels.append(row.pop(target_column))
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset {path} is empty.")

    feature_columns = [col for col in reader.fieldnames if col not in (target_column, ID_COLUMN)] if reader.fieldnames else None
    return rows, feature_columns, (labels if labels else None)


class FeatureTransformer:
    """Convert raw mushroom records into token features for Naive Bayes."""

    def __init__(self, rows: Sequence[Dict[str, str]], columns: Sequence[str],
                 text_column: Optional[str] = TEXT_COLUMN, max_numeric_bins: int = 8) -> None:
        self.columns = list(columns)
        self.text_column = text_column if text_column in self.columns else None
        self.max_numeric_bins = max_numeric_bins
        self.numeric_bins: Dict[str, List[float]] = {}
        self.numeric_stats: Dict[str, Dict[str, float]] = {}
        self.categorical_columns: List[str] = []
        self.word_pattern = re.compile(r"[a-z0-9]+", re.IGNORECASE)
        self.text_unigram_vocab: set[str] = set()
        self.text_bigram_vocab: set[str] = set()

        self._analyze_columns(rows)
        if self.text_column:
            self._build_text_vocabulary(rows)

    def _analyze_columns(self, rows: Sequence[Dict[str, str]]) -> None:
        for col in self.columns:
            if col == self.text_column:
                continue

            values = [rows[i].get(col, "") for i in range(len(rows))]
            numeric_values: List[float] = []
            valid_numeric = 0
            for value in values:
                value = value.strip()
                if not value:
                    continue
                try:
                    numeric_values.append(float(value))
                    valid_numeric += 1
                except ValueError:
                    continue

            if numeric_values and valid_numeric >= max(4, int(0.6 * len(values))):
                thresholds = self._compute_bins(numeric_values, self.max_numeric_bins)
                self.numeric_bins[col] = thresholds
                self.numeric_stats[col] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values)
                }
            else:
                self.categorical_columns.append(col)

    def _compute_bins(self, values: Sequence[float], max_bins: int) -> List[float]:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        thresholds: List[float] = []
        if n <= 1:
            return thresholds

        for i in range(1, max_bins):
            idx = int(round(i * n / max_bins))
            if idx <= 0 or idx >= n:
                continue
            candidate = sorted_vals[idx]
            if thresholds and candidate <= thresholds[-1]:
                continue
            thresholds.append(candidate)
        return thresholds

    def _build_text_vocabulary(self, rows: Sequence[Dict[str, str]]) -> None:
        unigram_counter: Counter[str] = Counter()
        bigram_counter: Counter[str] = Counter()

        for row in rows:
            raw_text = row.get(self.text_column, "") if self.text_column else ""
            words = self._basic_tokenize(raw_text)
            if not words:
                continue
            unigram_counter.update(words)
            bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
            bigram_counter.update(bigrams)

        if not unigram_counter:
            return

        min_freq = 2 if len(rows) > 30 else 1
        self.text_unigram_vocab = {word for word, count in unigram_counter.items() if count >= min_freq}
        self.text_bigram_vocab = {bg for bg, count in bigram_counter.items() if count >= min_freq}

        # Limit vocabulary sizes to keep the model efficient.
        max_unigrams = 800
        max_bigrams = 400
        if len(self.text_unigram_vocab) > max_unigrams:
            self.text_unigram_vocab = set(word for word, _ in unigram_counter.most_common(max_unigrams))
        if len(self.text_bigram_vocab) > max_bigrams:
            self.text_bigram_vocab = set(bg for bg, _ in bigram_counter.most_common(max_bigrams))

    def transform(self, row: Dict[str, str]) -> List[str]:
        tokens: List[str] = []
        for col in self.columns:
            if col == self.text_column:
                continue
            tokens.extend(self._transform_column(col, row.get(col, "")))

        if self.text_column:
            tokens.extend(self._transform_text(row.get(self.text_column, "")))
        return tokens

    def _transform_column(self, col: str, value: str) -> List[str]:
        tokens: List[str] = []
        raw_value = value.strip()
        if col in self.numeric_bins:
            if not raw_value:
                return [f"{col}__missing"]
            try:
                numeric_value = float(raw_value)
            except ValueError:
                normalized = self._normalize_categorical(raw_value)
                return [f"{col}__non_numeric_{normalized}", f"{col}__present"]

            bin_index = self._assign_bin(col, numeric_value)
            tokens.append(f"{col}__bin_{bin_index}")
            tokens.append(f"{col}__round_{numeric_value:.1f}")
            scale = int(numeric_value // 5)
            tokens.append(f"{col}__scale_{scale}")
            tokens.append(f"{col}__present")
        else:
            if not raw_value:
                tokens.append(f"{col}__missing")
            else:
                normalized = self._normalize_categorical(raw_value)
                tokens.append(f"{col}__value_{normalized}")
                tokens.append(f"{col}__present")
        return tokens

    def _assign_bin(self, col: str, value: float) -> int:
        thresholds = self.numeric_bins.get(col, [])
        index = 0
        while index < len(thresholds) and value > thresholds[index]:
            index += 1
        return index

    def _normalize_categorical(self, value: str) -> str:
        lowered = value.lower()
        cleaned_chars: List[str] = []
        for ch in lowered:
            if ch.isalnum():
                cleaned_chars.append(ch)
            elif ch in {"#", "-", "_"}:
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        normalized = "_".join(part for part in "".join(cleaned_chars).split())
        return normalized if normalized else "symbolic"

    def _basic_tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in self.word_pattern.findall(text or "")]

    def _transform_text(self, text: str) -> List[str]:
        if not text:
            return ["desc__missing"]
        words = self._basic_tokenize(text)
        if not words:
            return ["desc__missing"]

        tokens: List[str] = ["desc__present"]
        unigram_tokens = 0
        for word in words:
            if word in self.text_unigram_vocab:
                tokens.append(f"desc__uni_{word}")
                unigram_tokens += 1
        bigram_tokens = 0
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i + 1]}"
            if bigram in self.text_bigram_vocab:
                tokens.append(f"desc__bi_{bigram}")
                bigram_tokens += 1
        if unigram_tokens == 0 and bigram_tokens == 0:
            tokens.append("desc__rare")
        tokens.append(f"desc__length_bucket_{len(words) // 10}")
        return tokens


class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier for tokenized features."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Dict[str, int] = defaultdict(int)
        self.vocabulary: set[str] = set()
        self.total_documents: int = 0

    def fit(self, dataset: Sequence[Tuple[Iterable[str], str]]) -> None:
        self.class_counts.clear()
        self.token_counts = defaultdict(Counter)
        self.total_tokens.clear()
        self.vocabulary.clear()
        self.total_documents = len(dataset)

        for tokens, label in dataset:
            self.class_counts[label] += 1
            for token in tokens:
                self.token_counts[label][token] += 1
                self.total_tokens[label] += 1
                self.vocabulary.add(token)

    def predict_one(self, tokens: Iterable[str]) -> str:
        if not self.class_counts:
            raise RuntimeError("Classifier has not been fitted yet.")

        vocab_size = max(len(self.vocabulary), 1)
        class_prob_denominator = self.total_documents + self.alpha * len(self.class_counts)
        best_label = None
        best_log_prob = float("-inf")

        for label, class_count in self.class_counts.items():
            log_prob = math.log((class_count + self.alpha) / class_prob_denominator)
            token_total = self.total_tokens[label]
            token_denom = token_total + self.alpha * vocab_size
            for token in tokens:
                token_count = self.token_counts[label].get(token, 0)
                log_prob += math.log(token_count + self.alpha) - math.log(token_denom)
            if log_prob > best_log_prob or best_label is None:
                best_label = label
                best_log_prob = log_prob
        if best_label is None:
            # Fall back to majority class in extremely unlikely edge cases.
            best_label = max(self.class_counts.items(), key=lambda item: item[1])[0]
        return best_label

    def predict(self, dataset: Sequence[Iterable[str]]) -> List[str]:
        return [self.predict_one(tokens) for tokens in dataset]


def stratified_k_fold_indices(labels: Sequence[str], k: int, seed: int) -> List[List[int]]:
    by_class: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    folds: List[List[int]] = [[] for _ in range(k)]
    for class_indices in by_class.values():
        for offset, idx in enumerate(class_indices):
            folds[offset % k].append(idx)

    for fold in folds:
        fold.sort()
    return folds


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    classes = sorted(set(y_true))
    if not classes:
        return 0.0
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        if tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1_sum += f1
    return f1_sum / len(classes)


def perform_cross_validation(rows: Sequence[Dict[str, str]], labels: Sequence[str], columns: Sequence[str],
                             folds: int, seed: int) -> List[float]:
    fold_indices = stratified_k_fold_indices(labels, folds, seed)
    scores: List[float] = []

    for fold_id, validation_indices in enumerate(fold_indices):
        validation_index_set = set(validation_indices)
        train_indices = [idx for idx in range(len(rows)) if idx not in validation_index_set]

        train_rows = [rows[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]
        validation_rows = [rows[idx] for idx in validation_indices]
        validation_labels = [labels[idx] for idx in validation_indices]

        transformer = FeatureTransformer(train_rows, columns)
        train_dataset = [
            (transformer.transform(rows[idx]), labels[idx])
            for idx in train_indices
        ]

        classifier = NaiveBayesClassifier(alpha=0.5)
        classifier.fit(train_dataset)

        validation_tokens = [transformer.transform(row) for row in validation_rows]
        predictions = classifier.predict(validation_tokens)
        score = macro_f1_score(validation_labels, predictions)
        scores.append(score)
        print(f"Fold {fold_id + 1}/{folds} macro F1: {score:.4f}")

    return scores


def save_submission(path: str, rows: Sequence[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([ID_COLUMN, TARGET_COLUMN])
        for row_id, label in rows:
            writer.writerow([row_id, label])


def main() -> None:
    random.seed(RANDOM_SEED)

    train_rows, feature_columns, train_labels = load_dataset("train.csv", target_column=TARGET_COLUMN)
    if feature_columns is None or train_labels is None:
        raise RuntimeError("Failed to load training data.")

    folds = 5 if len(train_rows) >= 50 else max(2, len(train_rows))
    print(f"Performing {folds}-fold stratified cross-validation...")
    cv_scores = perform_cross_validation(train_rows, train_labels, feature_columns, folds, RANDOM_SEED)
    mean_f1 = sum(cv_scores) / len(cv_scores)
    print(f"Cross-validated macro F1: {mean_f1:.4f}")

    print("Training final model on full dataset...")
    final_transformer = FeatureTransformer(train_rows, feature_columns)
    training_dataset = [
        (final_transformer.transform(row), label)
        for row, label in zip(train_rows, train_labels)
    ]

    classifier = NaiveBayesClassifier(alpha=0.5)
    classifier.fit(training_dataset)

    test_rows, _, _ = load_dataset("test.csv")
    predictions: List[Tuple[str, str]] = []
    for row in test_rows:
        tokens = final_transformer.transform(row)
        predicted_label = classifier.predict_one(tokens)
        row_id = row.get(ID_COLUMN, "")
        predictions.append((row_id, predicted_label))

    submission_path = os.path.join(os.getcwd(), "submission.csv")
    save_submission(submission_path, predictions)
    print(f"Generated {submission_path} with {len(predictions)} entries.")


if __name__ == "__main__":
    main()
