import numpy as np
import torch
from gensim.models import Word2Vec
from collections import defaultdict
from itertools import product
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import textacy
from tqdm import tqdm
import json
import pickle as pkl
import numpy as np
from scipy.sparse import coo_matrix


