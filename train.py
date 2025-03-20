from preprocessing import PreprocessingPipeline
from mertics import MetricComputer
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

class TrainingPipeline:
    def __init__(self, model, dataset):
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

        self.preprocessor = PreprocessingPipeline()
        self.dataset = dataset.map(
            self.preprocessor.tokenize_and_align_labels,
            batched=True
        )
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.preprocessor.tokenizer)

        self.compute_metrics = MetricComputer().compute_metrics

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.preprocessor.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

    def train(self):
        self.trainer.train()
