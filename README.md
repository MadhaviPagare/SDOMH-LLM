
# ClinicalLongformer - README

## Introduction
This notebook involves using Clinical Longformer, a specialized transformer-based model for clinical text analysis. It appears to be used for tasks like named entity recognition, classification, or another NLP use case within the clinical domain.
## Dependencies
The following dependencies are commonly used with Clinical Longformer:
- Python 3.x
- Transformers Library (HuggingFace)
- PyTorch
- Pandas
- NumPy

Please refer to the notebook for a complete list of required libraries.
## Usage
To use this notebook, make sure to install all the necessary dependencies and set up a suitable Python environment. Load the notebook and follow the instructions to execute each cell in sequence.

Code Snippet:
!export CUDA_VISIBLE_DEVICES=0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Now import your GPU-related libraries, such as PyTorch
import torch
torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)
# Import libraries and set up CUDA
import os
os.environ["CUDA_V...

Code Snippet:
1.#ORIGINAL TEST FINAL as policy P1

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, LongformerModel, LongformerConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqd...

Code Snippet:
2.#Testing on Human annotated Labels policy P2

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tq...

Code Snippet:
3.#Testing on ChatGPT equivalent texts policy P3

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_sele...

Code Snippet:
...


## Contributing
If you'd like to contribute, please fork the repository, create a new branch, make your changes, and submit a pull request for review.
## License
This project is licensed under the MIT License. See the LICENSE file for details.
