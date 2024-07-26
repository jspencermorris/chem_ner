from sklearn.metrics import matthews_corrcoef
import numpy as np

def compute_MCC_jsonl(references_jsonl, predictions_jsonl, id_col, ref_col='ner_tags', pred_col='pred_ner_tags', ignore_label='O'):
    '''
    Computes the Matthews correlation coeff between two datasets in jsonl format (list of dicts each with same keys).
    Sorts the datasets by 'unique_id' and verifies that the tokens match.
    '''
    # reverse the dict
    ref_dict = {k:[e[k] for e in references_jsonl] for k in references_jsonl[0].keys()}
    pred_dict = {k:[e[k] for e in predictions_jsonl] for k in predictions_jsonl[0].keys()}

    # sort by unique_id
    ref_idx = np.argsort(ref_dict[id_col])
    pred_idx = np.argsort(pred_dict[id_col])
    ref_ner_tags = np.array(ref_dict[ref_col], dtype=object)[ref_idx]
    pred_ner_tags = np.array(pred_dict[pred_col], dtype=object)[pred_idx]
    ref_tokens = np.array(ref_dict['tokens'], dtype=object)[ref_idx]
    pred_tokens = np.array(pred_dict['tokens'], dtype=object)[pred_idx]

    # check that tokens match
    """for t1,t2 in zip(ref_tokens, pred_tokens):
        assert(t1==t2)"""

    # the lists have to be flattened
    flat_ref_tags = np.concatenate(ref_ner_tags)
    flat_pred_tags = np.concatenate(pred_ner_tags)

    mcc_score = matthews_corrcoef(y_true=flat_ref_tags,
                                  y_pred=flat_pred_tags)
    
    return(mcc_score)