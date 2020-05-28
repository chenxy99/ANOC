from typing import List, Dict

from mypy_extensions import TypedDict
import numpy as np
import torch

# Type hint for objects returned by ``TrainingDatasetWithPenultimateSaliencySCST.__getitem__``.
TrainingInstanceWithPenultimateSaliencySCST = TypedDict(
    "TrainingInstanceWithPenultimateSaliencySCST",
    {"image_id": int, "image_features": np.ndarray, "penultimate_features": np.ndarray, "captions": List[str]},
)

# Type hint for objects returned by ``TrainingDatasetWithPenultimateSaliencySCST.collate_fn``.
TrainingBatchWithPenultimateSaliencySCST = TypedDict(
    "TrainingBatchWithPenultimateSaliencySCST",
    {
        "image_id": torch.LongTensor,
        "image_features": torch.FloatTensor,
        "penultimate_features": torch.FloatTensor,
        "captions": List[List[str]],
    },
)

# Type hint for objects returned by ``TrainingEvaluationDatasetWithConstraintsSCST.__getitem__``.
TrainingEvaluationInstanceWithConstraintsAndPenultimateSaliencySCST = TypedDict(
    "TrainingEvaluationInstanceWithConstraintsAndPenultimateSaliencySCST",
    {
        "image_id": int,
        "image_features": np.ndarray,
        "penultimate_features": np.ndarray,
        "captions": List[str],
        "fsm": torch.ByteTensor,
        "num_states": int,
        "num_constraints": int,
     },
)

# Type hint for objects returned by ``TrainingEvaluationDatasetWithConstraintsSCST.collate_fn``.
TrainingEvaluationBatchWithConstraintsAndPenultimateSaliencySCST = TypedDict(
    "TrainingEvaluationBatchWithConstraintsAndPenultimateSaliencySCST",
    {
        "image_id": torch.LongTensor,
        "image_features": torch.FloatTensor,
        "penultimate_features": torch.FloatTensor,
        "captions": List[List[str]],
        "fsm": torch.ByteTensor,
        "num_constraints": torch.LongTensor,
    },
)

###########################################################################################
# Type hint for objects returned by ``TrainingDatasetWithPenultimateSaliency.__getitem__``.
TrainingInstanceWithPenultimateSaliency = TypedDict(
    "TrainingInstanceWithPenultimateSaliency",
    {"image_id": int, "image_features": np.ndarray, "penultimate_features": np.ndarray, "caption_tokens": List[int]},
)

# Type hint for objects returned by ``TrainingDatasetWithPenultimateSaliency.collate_fn``.
TrainingBatchWithPenultimateSaliency = TypedDict(
    "TrainingBatchWithPenultimateSaliency",
    {
        "image_id": torch.LongTensor,
        "image_features": torch.FloatTensor,
        "penultimate_features": torch.FloatTensor,
        "caption_tokens": torch.LongTensor,
    },
)

# Type hint for objects returned by ``EvaluationDatasetWithPenultimateSaliency.__getitem__``.
EvaluationInstanceWithPenultimateSaliency = TypedDict(
    "EvaluationInstance", {"image_id": int, "image_features": np.ndarray, "penultimate_features": np.ndarray}
)
EvaluationInstanceWithConstraintsAndPenultimateSaliency = TypedDict(
    "EvaluationInstanceWithConstraintsAndPenultimateSaliency",
    {
        "image_id": int,
        "image_features": np.ndarray,
        "penultimate_features": np.ndarray,
        "fsm": torch.ByteTensor,
        "num_states": int,
        "num_constraints": int,
    },
)

# Type hint for objects returned by ``EvaluationDatasetWithPenultimateSaliency.collate_fn``.
EvaluationBatchWithPenultimateSaliency = TypedDict(
    "EvaluationDatasetWithPenultimateSaliency", {"image_id": torch.LongTensor, "image_features": torch.FloatTensor,
                                                 "penultimate_features": torch.FloatTensor}
)
EvaluationBatchWithConstraintsAndPenultimateSaliency = TypedDict(
    "EvaluationBatchWithConstraintsAndPenultimateSaliency",
    {
        "image_id": int,
        "image_features": torch.FloatTensor,
        "penultimate_features": torch.FloatTensor,
        "fsm": torch.ByteTensor,
        "num_constraints": torch.LongTensor,
    },
)

ConstraintBoxes = TypedDict(
    "ConstraintBoxes", {"boxes": np.ndarray, "class_names": List[str], "score": np.ndarray}
)

Prediction = TypedDict("Prediction", {"image_id": int, "caption": str})
