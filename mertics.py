import numpy as np
from evaluate import load

class MetricComputer:
    def __init__(self):
        self.metric = load('seqeval')
    
    def compute_metrics(self, p, label_list):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []
        for predictions, label in zip(predictions, labels):
            pred_tags = []
            true_tag = []
            for p_val, l_val in zip(predictions, label):
                if l_val != -100:
                    pred_tags.append(label_list[p_val])
                    true_tag.append(label_list[l_val])
            true_predictions.append(pred_tags)
            true_labels.append(true_tag)
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
