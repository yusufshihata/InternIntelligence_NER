from transformers import (
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

class TrainingPipeline:
    def __init__(self):
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
        )


