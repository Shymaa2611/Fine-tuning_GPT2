import csv
import os
import ast
from pathlib import Path
import random


def create_directories():
  train_path = Path('data/train')
  test_path = Path('data/test')
  for path in [train_path, test_path]:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        (path / f'{i} star').mkdir(exist_ok=True)
  return train_path,test_path

def save_to_file(folder, label, content, idx):
    folder_path = folder / f'{label} star'
    file_path = folder_path / f'document_{idx}.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def read_dataset(dataset_path):
 with open(dataset_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    data = []
    for row in reader:
        label_dict = ast.literal_eval(row['label'])
        label = label_dict['label']
        content = row['content']
        data.append((label, content))
    return data

def create_dataset(data_path):
  train_path,test_path=create_directories()
  data=read_dataset(data_path)
  random.shuffle(data)
  split_ratio = 0.8
  split_index = int(len(data) * split_ratio)
  train_data = data[:split_index]
  test_data = data[split_index:]
  for idx, (label, content) in enumerate(train_data):
    label_num = label.split()[0]  
    save_to_file(train_path, label_num, content, idx)
  for idx, (label, content) in enumerate(test_data):
    label_num = label.split()[0]  
    save_to_file(test_path, label_num, content, idx)


if __name__=="__main__":
   data_path="dataset.csv"
   create_dataset(data_path)
