import torch
from model import Gpt2ClassificationCollator
from torch.utils.data import DataLoader
from dataset import myDataset
from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from ml_things import plot_dict
import tqdm
from sklearn.metrics import accuracy_score
from train import train
from validation import validation
from utils import save_checkpoint

labels_ids = {'1 star': 0, '2 star': 1,'3 star':2,'4 star':3,'5 star':4}

def get_model():
    n_labels = len(labels_ids)
    model_config = GPT2Config.from_pretrained(pretrained_model_name='gpt2', num_labels=n_labels)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name='gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    return model,tokenizer


def run():
    batch_size = 32
    max_length = 60
    epochs=5
    model,tokenizer=get_model()
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)
    train_dataset = myDataset(path='data/train', 
                               use_tokenizer=tokenizer)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    valid_dataset =  myDataset(path='/data/test', 
                               use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                  )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    print('Epoch')
    for epoch in tqdm(range(epochs)):
      train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler,model)
      train_acc = accuracy_score(train_labels, train_predict)
      valid_labels, valid_predict, val_loss = validation(valid_dataloader,model)
      val_acc = accuracy_score(valid_labels, valid_predict)
      print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
      checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
      save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)
      all_loss['train_loss'].append(train_loss)
      all_loss['val_loss'].append(val_loss)
      all_acc['train_acc'].append(train_acc)
      all_acc['val_acc'].append(val_acc)
      plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
      plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])


if __name__=="__main__":
   run()