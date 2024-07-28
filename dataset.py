import io
import os
import tqdm
from torch.utils.data import Dataset
from ml_things import  fix_text


class myDataset(Dataset):
  
  def __init__(self, path, use_tokenizer):
    if not os.path.isdir(path):
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    for label in ['1 star','2 star','3 star','4 star','5 star']:
      sentiment_path = os.path.join(path, label)
      files_names = os.listdir(sentiment_path)
      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        content = fix_text(content)
        self.texts.append(content)
        self.labels.append(label)
    self.n_examples = len(self.labels)
    

    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}

