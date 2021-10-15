from .data_collator import DataCollatorForWholeWordMask, DataCollatorForLanguageModelingWithElectra
from .language_modeling import (
    TextDatasetForNextSentencePrediction, LineByLineTextDataset,
    HFTextDatasetForNextSentencePrediction, HFLineByLineTextDataset
)
from .model import ElectraForPretrainingModel
from .trainer import MyTrainer
from .training_args import _setup_devices
