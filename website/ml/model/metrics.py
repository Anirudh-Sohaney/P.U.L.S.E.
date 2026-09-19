"""Model metrics for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass, field


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    classification_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Model Performance:\n"
            f"  Accuracy:  {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1 Score:  {self.f1_score:.4f}\n"
            f"  ROC AUC:   {self.roc_auc:.4f}\n"
            f"\n"
            f"Confusion Matrix:\n"
            f"  True Positives:  {self.true_positives}\n"
            f"  True Negatives:  {self.true_negatives}\n"
            f"  False Positives: {self.false_positives}\n"
            f"  False Negatives: {self.false_negatives}"
        )
