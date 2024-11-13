import pandas as pd

splits = {'train': 'cifar100/train-00000-of-00001.parquet', 'test': 'cifar100/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/uoft-cs/cifar100/" + splits["train"])

