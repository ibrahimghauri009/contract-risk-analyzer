"""Supervised Clause Classification and Risk Flagging Model.

Trains and evaluates a fast discriminative classifier on CUAD clause embeddings
to predict clause types and probability confidence scores with real precision/recall metrics.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# High-impact legal risk categories for focused training & evaluation
RISK_CATEGORIES = [
    "Cap On Liability",
    "Uncapped Liability",
    "Non-Compete",
    "Termination For Convenience",
    "Exclusivity",
    "Governing Law",
    "Audit Rights",
    "Insurance",
    "Anti-Assignment",
    "Liquidated Damages",
    "License Grant",
    "Post-Termination Services",
    "Revenue/Profit Sharing",
    "Warranty Duration",
    "Covenant Not To Sue"
]

class ClauseClassifier:
    """Supervised classifier trained on CUAD clause embeddings."""

    def __init__(
        self,
        model_path: Path = settings.CLASSIFIER_PATH,
        encoder_path: Path = settings.LABEL_ENCODER_PATH,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME
    ):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.embedding_model_name = embedding_model_name
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.classifier: Optional[LogisticRegression] = None
        self.label_encoder: Optional[LabelEncoder] = None

        if self.model_path.exists() and self.encoder_path.exists():
            self.load()

    def train(
        self,
        data_csv: Path = settings.PROCESSED_DATA_DIR / "cuad_clauses.csv",
        max_samples_per_category: int = 120,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Trains a Logistic Regression classifier over clause embeddings.
        Evaluates on held-out test split and returns metrics dictionary.
        """
        logger.info(f"Loading training data from {data_csv}...")
        df = pd.read_csv(data_csv)

        # Filter for selected risk categories
        filtered_df = df[df["category"].isin(RISK_CATEGORIES)].copy()
        filtered_df = filtered_df.dropna(subset=["clause_text"])
        filtered_df["clause_text"] = filtered_df["clause_text"].astype(str)

        # Truncate clauses to 512 chars for fast, focused semantic representation
        filtered_df["clause_text"] = filtered_df["clause_text"].apply(lambda t: t[:512].strip())

        # Balance classes: sample up to max_samples_per_category per category
        balanced_dfs = []
        for cat, group in filtered_df.groupby("category"):
            n_samples = min(len(group), max_samples_per_category)
            balanced_dfs.append(group.sample(n=n_samples, random_state=random_state))
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Prepared balanced dataset: {len(balanced_df)} samples across {balanced_df['category'].nunique()} categories.")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(balanced_df["category"])

        # Convert to standard Python lists
        texts_list = balanced_df["clause_text"].tolist()
        y_array = np.array(y_encoded)

        # Train/Test Split (Stratified 80/20)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts_list,
            y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array
        )

        logger.info(f"Generating dense embeddings for {len(X_train_text)} training samples...")
        X_train_emb = self.embedder.encode(
            X_train_text,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Generating dense embeddings for {len(X_test_text)} testing samples...")
        X_test_emb = self.embedder.encode(
            X_test_text,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        logger.info("Training Logistic Regression classifier (C=2.0, class_weight='balanced')...")
        self.classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
        self.classifier.fit(X_train_emb, y_train)

        # Evaluation on held-out test set
        y_pred = self.classifier.predict(X_test_emb)
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        report_text = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        accuracy = float(np.mean(y_test == y_pred))

        logger.info("\n" + "="*60 + "\nCLASSIFICATION BENCHMARK REPORT (HELD-OUT TEST SET)\n" + "="*60)
        logger.info(f"\n{report_text}")
        logger.info(f"Macro F1 Score: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Accuracy: {accuracy:.4f}")

        # Save artifacts
        self.save()

        metrics = {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "accuracy": accuracy,
            "report_text": report_text,
            "report_dict": report_dict,
            "categories": list(self.label_encoder.classes_),
            "train_size": len(X_train_text),
            "test_size": len(X_test_text)
        }
        return metrics

    def predict(self, text: str, threshold: float = 0.20) -> Dict[str, Any]:
        """
        Predicts clause category with probability confidence for a given clause text snippet.
        """
        if self.classifier is None or self.label_encoder is None:
            raise ValueError("Classifier is not trained or loaded. Call train() or load() first.")

        embedding = self.embedder.encode([text[:512]], normalize_embeddings=True)
        probabilities = self.classifier.predict_proba(embedding)[0]
        
        top_idx = int(np.argmax(probabilities))
        top_confidence = float(probabilities[top_idx])
        top_category = str(self.label_encoder.classes_[top_idx])

        # Get top-3 candidate categories
        sorted_indices = np.argsort(probabilities)[::-1][:3]
        top_candidates = [
            {"category": str(self.label_encoder.classes_[i]), "confidence": float(probabilities[i])}
            for i in sorted_indices
        ]

        is_confident = top_confidence >= threshold

        return {
            "predicted_category": top_category if is_confident else "Uncategorized/General",
            "confidence": top_confidence,
            "is_confident": is_confident,
            "top_candidates": top_candidates
        }

    def save(self):
        """Saves trained model and label encoder to disk."""
        logger.info(f"Saving classifier to {self.model_path}...")
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        logger.info("Model artifacts saved successfully.")

    def load(self):
        """Loads trained model and label encoder from disk."""
        if self.model_path.exists() and self.encoder_path.exists():
            self.classifier = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.encoder_path)
            logger.info(f"Loaded classifier with {len(self.label_encoder.classes_)} classes.")
        else:
            raise FileNotFoundError(f"Model artifacts not found at {self.model_path}")

if __name__ == "__main__":
    classifier = ClauseClassifier()
    metrics = classifier.train()
    print("\n--- Live Test Prediction ---")
    sample = "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES."
    pred = classifier.predict(sample)
    print(f"Sample Clause: '{sample}'")
    print(f"Prediction: {pred['predicted_category']} (Confidence: {pred['confidence']:.2%})")
