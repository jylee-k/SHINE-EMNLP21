- Check understanding of each baseline model
    - TextGCN - [Paper](https://arxiv.org/abs/1809.05679) [GitHub](https://github.com/codeKgu/Text-GCN)
    - TensorGCN - [Paper](https://arxiv.org/abs/2001.05313) [GitHub](https://github.com/THUMLP/TensorGCN_pytorch)
    - GFN - [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705121009217) (No code)
    - HeteGCN - [Paper](https://arxiv.org/abs/2008.12842) [GitHub](https://github.com/microsoft/HeteGCN)
    - SHINE - [Paper](https://aclanthology.org/2021.emnlp-main.247/) [GitHub](https://github.com/tata1661/SHINE-EMNLP21)

- updated 26 May 2024 for newer models to benchmark against
    - DADGNN - [Paper](https://aclanthology.org/2021.emnlp-main.642/) [GitHub](https://github.com/KEAML-JLU/DADGNN) (DGL issues; hold first)
    - Text-Level GNN - [Paper](https://aclanthology.org/2021.emnlp-main.642/) [GitHub](https://github.com/Cynwell/Text-Level-GNN)
    - HyperGAT - [Paper](https://aclanthology.org/2020.emnlp-main.399/) [GitHub](https://github.com/kaize0409/HyperGAT_TextClassification)

- updated 16 June 2024 for newer / good quality repos for ease of replication
    - MEGCN - [Paper](https://arxiv.org/abs/2204.04618) [GitHub](https://github.com/usydnlp/ME_GCN)
    - SGC - [Paper](https://arxiv.org/abs/1902.07153) [GitHub](https://github.com/Tiiiger/SGC)

- Preferrably we need all of them compatible with the latest versions of the dependencies at Python 3.11

- Datasets
    - The datasets are available under ```preprocess``` as ```xxx_split.json```
    - TREC/ARC/Bloom’s taxonomy/MedQUAD → help to do EDA
        - EDA required: 
            - label distribution
            - question length distribution (by word)
            - any other EDA deemed fit


            