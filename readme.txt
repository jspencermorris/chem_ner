Evaluating BERT Variants and Classification Approaches for Chemical Named Entity Recognition

Aug 2024

Fine-tuned transformer-based models on chemistry-related text to devise models for chemical entity identification and relation extraction in academic and industrial patent literature.

Virtual environment details in ./requirements.txt.

Datasets (original) stored locally, but primary transformed tabular data is available in ./data/interim/.

Data preprocessing and modeling done using Jupyter Notebooks.  Details in ./notebooks/. The notebooks included are:
    CLUB_NER_A_Baselines_and_BERT_Prototype.ipynb
    CLUB_NER_B_baseBERT_Experiments.ipynb
    CLUB_NER_C_SciBERT_Experiments.ipynb
    CLUB_NER_D_SpanBERT_Experiments.ipynb
    CLUB_NER_E_Ensembling_and_More_Analysis.ipynb

Finalized tuned Models for tabular data were pickled and are available in ./models/.

Key References in ./references/literature/.

Final and interim reports in ./reports/.