from __future__ import annotations
import numpy as np
from scipy import linalg
from scipy.special import softmax
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import warnings


class LogicalVarianceWarning(RuntimeWarning):
    pass


@dataclass
class StatisticalProfile:
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    sample_count: int
    eigenvalues: np.ndarray = field(init=False)
    condition_number: float = field(init=False)

    def __post_init__(self):
        self.eigenvalues = np.linalg.eigvalsh(self.covariance)
        self.condition_number = float(
            np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-12)
        )


class AbstractValidator(metaclass=ABCMeta):
    @abstractmethod
    def ingest(self, vector: np.ndarray) -> None: ...

    @abstractmethod
    def evaluate(self, vector: np.ndarray) -> Dict[str, Any]: ...


class MahalanobisOODDetector(AbstractValidator):
    def __init__(
        self, threshold: float = 3.0, min_samples: int = 1, regularization: float = 1e-6
    ):
        self.threshold = threshold
        self.min_samples = min_samples
        self.regularization = regularization
        self._running_sum: Optional[np.ndarray] = None
        self._running_outer: Optional[np.ndarray] = None
        self._count: int = 0
        self._profile: Optional[StatisticalProfile] = None
        self._buffer: List[np.ndarray] = []

    def ingest(self, vector: np.ndarray) -> None:
        vector = vector.flatten().astype(np.float64)
        self._buffer.append(vector)
        if self._running_sum is None:
            self._running_sum = np.zeros_like(vector)
            self._running_outer = np.zeros((vector.size, vector.size))
        self._running_sum += vector
        self._running_outer += np.outer(vector, vector)
        self._count += 1
        if self._count >= self.min_samples:
            self._recompute_profile()

    def _recompute_profile(self) -> None:
        mean = self._running_sum / self._count
        covariance = (self._running_outer / self._count) - np.outer(mean, mean)
        covariance += np.eye(covariance.shape[0]) * self.regularization
        try:
            precision = linalg.inv(covariance)
        except linalg.LinAlgError:
            precision = linalg.pinv(covariance)
        self._profile = StatisticalProfile(
            mean=mean,
            precision=precision,
            covariance=covariance,
            sample_count=self._count,
        )

    def compute_distance(self, vector: np.ndarray) -> float:
        if self._profile is None:
            return float("inf")
        delta = vector.flatten().astype(np.float64) - self._profile.mean
        distance = float(
            np.sqrt(np.clip(delta @ self._profile.precision @ delta, 0, None))
        )
        return distance

    def is_out_of_distribution(self, vector: np.ndarray) -> bool:
        return self.compute_distance(vector) > self.threshold

    def evaluate(self, vector: np.ndarray) -> Dict[str, Any]:
        dist = self.compute_distance(vector)
        return {
            "mahalanobis_distance": dist,
            "is_ood": dist > self.threshold,
            "profile_condition": (
                self._profile.condition_number if self._profile else None
            ),
        }


class EntropyValidator(AbstractValidator):
    def __init__(self, entropy_threshold: float = 4.0, base: float = 2.0):
        self.entropy_threshold = entropy_threshold
        self.base = base
        self._label_history: List[int] = []
        self._entropy_history: List[float] = []

    def calculate_shannon_entropy(self, probabilities: np.ndarray) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = probabilities[probabilities > 1e-12]
        probabilities = probabilities / np.sum(probabilities)
        log_probs = np.log(probabilities) / np.log(self.base)
        return float(-np.sum(probabilities * log_probs))

    def calculate_renyi_entropy(
        self, probabilities: np.ndarray, alpha: float = 2.0
    ) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = probabilities[probabilities > 1e-12]
        probabilities = probabilities / np.sum(probabilities)
        if alpha == 1.0:
            return self.calculate_shannon_entropy(probabilities)
        return float(
            np.log(np.sum(probabilities**alpha)) / ((1.0 - alpha) * np.log(self.base))
        )

    def validate_predictions(self, prediction_logits: np.ndarray) -> Tuple[bool, float]:
        probabilities = softmax(prediction_logits)
        entropy = self.calculate_shannon_entropy(probabilities)
        self._entropy_history.append(entropy)
        if entropy > self.entropy_threshold:
            warnings.warn("LogicalVarianceWarning triggered", LogicalVarianceWarning)
        return entropy <= self.entropy_threshold, entropy

    def ingest(self, vector: np.ndarray) -> None:
        pass

    def evaluate(self, vector: np.ndarray) -> Dict[str, Any]:
        probs = softmax(vector)
        entropy = self.calculate_shannon_entropy(probs)
        return {
            "entropy": entropy,
            "is_stable": entropy <= self.entropy_threshold,
            "renyi_2": self.calculate_renyi_entropy(probs, 2.0),
        }


class ZScoreValidator(AbstractValidator):
    def __init__(self, z_threshold: float = 2.5, window_size: Optional[int] = None):
        self.z_threshold = z_threshold
        self.window_size = window_size
        self._values: List[float] = []
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._n: int = 0

    def ingest(self, vector: np.ndarray) -> None:
        scalar = float(np.mean(vector))
        self.ingest_scalar(scalar)

    def ingest_scalar(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2
        self._values.append(value)
        if self.window_size and len(self._values) > self.window_size:
            removed = self._values.pop(0)
            self._recompute_from_window()

    def _recompute_from_window(self) -> None:
        arr = np.array(self._values, dtype=np.float64)
        self._mean = float(np.mean(arr))
        self._m2 = float(np.sum((arr - self._mean) ** 2))
        self._n = len(self._values)

    def compute_zscore(self, value: float) -> float:
        if self._n < 2:
            return 0.0
        variance = self._m2 / (self._n - 1)
        std = np.sqrt(variance) if variance > 0 else 0.0
        # Handle zero standard deviation - return 0 instead of NaN
        if std == 0.0:
            return 0.0
        return float((value - self._mean) / std)

    def is_anomalous(self, value: float) -> bool:
        return abs(self.compute_zscore(value)) > self.z_threshold

    def evaluate(self, vector: np.ndarray) -> Dict[str, Any]:
        scalar = float(np.mean(vector))
        z = self.compute_zscore(scalar)
        return {
            "z_score": z,
            "is_anomalous": abs(z) > self.z_threshold,
            "mean": self._mean,
            "std": np.sqrt(self._m2 / (self._n - 1)) if self._n > 1 else 0.0,
        }


class UnifiedValidator:
    def __init__(
        self,
        mahalanobis_threshold: float = 3.0,
        entropy_threshold: float = 4.0,
        z_threshold: float = 2.5,
        memory_engine: Optional[Any] = None,
    ):
        self.mahalanobis = MahalanobisOODDetector(
            threshold=mahalanobis_threshold, regularization=1e-6
        )
        self.entropy = EntropyValidator(entropy_threshold=entropy_threshold)
        self.zscore = ZScoreValidator(z_threshold=z_threshold)
        self.memory_engine = memory_engine

    async def _get_record_count(self) -> int:
        if self.memory_engine is None:
            return 0
        return await self.memory_engine.get_record_count()

    async def validate_tensor(
        self, tensor: np.ndarray, prediction_logits: np.ndarray
    ) -> Dict[str, Any]:
        flat = tensor.flatten().astype(np.float64)

        record_count = await self._get_record_count()
        is_cold_start = record_count < 3

        self.mahalanobis.ingest(flat)
        mahal_result = self.mahalanobis.evaluate(flat)
        entropy_result = self.entropy.evaluate(prediction_logits)
        self.zscore.ingest(flat)
        z_result = self.zscore.evaluate(flat)

        mahal_ood = mahal_result["is_ood"]
        entropy_unstable = not entropy_result["is_stable"]
        zscore_anomalous = z_result["is_anomalous"]

        anomaly_votes = sum([mahal_ood, entropy_unstable, zscore_anomalous])
        is_anomalous = (anomaly_votes >= 2) or (mahal_ood and entropy_unstable)

        if is_cold_start:
            confidence = 0.5
            is_anomalous = False
        else:
            mahal_capped = min(mahal_result["mahalanobis_distance"], 50.0)
            confidence = 1.0 / (
                1.0
                + 0.1 * mahal_capped
                + 0.05 * entropy_result["entropy"]
                + 0.2 * abs(z_result["z_score"])
            )

        return {
            "mahalanobis_distance": mahal_result["mahalanobis_distance"],
            "is_ood": mahal_result["is_ood"],
            "entropy": entropy_result["entropy"],
            "renyi_entropy": entropy_result["renyi_2"],
            "z_score": z_result["z_score"],
            "mean": z_result["mean"],
            "std": z_result["std"],
            "is_anomalous": is_anomalous,
            "confidence": confidence,
            "is_cold_start": is_cold_start,
            "record_count": record_count,
        }
