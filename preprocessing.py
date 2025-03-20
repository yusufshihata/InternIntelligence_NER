from transformers import AutoTokenizer

class PreprocessingPipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_and_align_labels(self, data):
        tokenized_inputs = self.tokenizer(
            data['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=25,
        )

        all_labels = []

        for i, labels in enumerate(data["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            all_labels.append(label_ids)
        tokenized_inputs['labels'] = all_labels
        return tokenized_inputs
