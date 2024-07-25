from seqeval.metrics import classification_report, accuracy_score
from seqeval.scheme import IOB2
import numpy as np

def compute_seqeval_jsonl(references_jsonl, predictions_jsonl, id_col, ref_col='ner_tags', pred_col='pred_ner_tags', ignore_label='O'):
    '''
    Computes the seqeval scores between two datasets loaded from jsonl (list of dicts with same keys).
    Sorts the datasets by 'unique_id' and verifies that the tokens match.
    Additionally, computes seqeval metrics with and without the 'ignore_label'.
    '''
    # Extract the tags and reverse the dict
    ref_dict = {k: [e[k] for e in references_jsonl] for k in references_jsonl[0].keys()}
    pred_dict = {k: [e[k] for e in predictions_jsonl] for k in predictions_jsonl[0].keys()}
        
    # Sort by unique_id
    ref_idx = np.argsort(ref_dict[id_col])
    pred_idx = np.argsort(pred_dict[id_col])
    ref_ner_tags = np.array(ref_dict[ref_col], dtype=object)[ref_idx]
    pred_ner_tags = np.array(pred_dict[pred_col], dtype=object)[pred_idx]
    ref_tokens = np.array(ref_dict['tokens'], dtype=object)[ref_idx]
    pred_tokens = np.array(pred_dict['tokens'], dtype=object)[pred_idx]

    # Check that tokens match
    assert((ref_tokens == pred_tokens).all())
    
    # Get report for all labels
    report = classification_report(y_true=ref_ner_tags, y_pred=pred_ner_tags, scheme=IOB2, output_dict=True)
    
    # Extract values we care about
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    seqeval_results = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "support": score["support"],
        }
        for type_name, score in report.items()
    }
    seqeval_results["overall_precision"] = overall_score["precision"]
    seqeval_results["overall_recall"] = overall_score["recall"]
    seqeval_results["overall_f1"] = overall_score["f1-score"]
    seqeval_results["overall_accuracy"] = accuracy_score(y_true=ref_ner_tags, y_pred=pred_ner_tags)    

    # Filter out the 'ignore_label'
    filtered_ref_ner_tags = []
    filtered_pred_ner_tags = []
    for ref_seq, pred_seq in zip(ref_ner_tags, pred_ner_tags):
        filtered_ref_seq = []
        filtered_pred_seq = []
        for ref_tag, pred_tag in zip(ref_seq, pred_seq):
            if ref_tag != ignore_label:
                filtered_ref_seq.append(ref_tag)
                filtered_pred_seq.append(pred_tag)
        if filtered_ref_seq:
            filtered_ref_ner_tags.append(filtered_ref_seq)
            filtered_pred_ner_tags.append(filtered_pred_seq)

    # Get report for filtered labels
    report_no_other = classification_report(y_true=filtered_ref_ner_tags, y_pred=filtered_pred_ner_tags, scheme=IOB2, output_dict=True)
    
    # Extract values for filtered results
    report_no_other.pop("macro avg")
    report_no_other.pop("weighted avg")
    overall_score_no_other = report_no_other.pop("micro avg")

    seqeval_results_no_other = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "support": score["support"],
        }
        for type_name, score in report_no_other.items()
    }
    seqeval_results_no_other["overall_precision"] = overall_score_no_other["precision"]
    seqeval_results_no_other["overall_recall"] = overall_score_no_other["recall"]
    seqeval_results_no_other["overall_f1"] = overall_score_no_other["f1-score"]
    seqeval_results_no_other["overall_accuracy"] = accuracy_score(y_true=filtered_ref_ner_tags, y_pred=filtered_pred_ner_tags)    

    return seqeval_results, seqeval_results_no_other
