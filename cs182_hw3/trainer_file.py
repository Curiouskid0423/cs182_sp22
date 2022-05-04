root_folder = "./"
import os
import sys
sys.path.append(root_folder)
from segtok import tokenizer
from tqdm import tqdm
from collections import Counter
import torch as th
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import json
from utils import validate_to_array
from language_model import LanguageModel

d_released, vocabulary = None, None

def numerize_sequence(tokenized):
    return [w2i.get(w, unkI) for w in tokenized]
def pad_sequence(numerized, pad_index, to_length):
    pad = numerized[:to_length]
    padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask

def numerized2text(numerized):
    """ 
    Converts an integer sequence in the vocabulary into a string corresponding to the title.
    """
    converted_string = " ".join([i2w[n] for n in numerized if n != padI])
    return converted_string

def build_batch(dataset, indices):
    """ Builds a batch of source and target elements from the dataset.
    
        Arguments:
            dataset: List[db_element] -- A list of dataset elements
            indices: List[int] -- A list of indices of the dataset to sample
        Returns:
            batch_input: List[List[int]] -- List of source sequences
            batch_target: List[List[int]] -- List of target sequences
            batch_target_mask: List[List[int]] -- List of target batch masks
    """
    batch = np.array(dataset)[indices]
    batch_numerized = np.array([b['numerized'] for b in batch])
    start_tokens = np.array([[startI] for _ in range(len(indices))])
    batch_input = np.concatenate((start_tokens, batch_numerized), axis=1)

    batch_input = batch_input[:, :-1]
    batch_target = batch_numerized
    batch_target_mask = np.array([a['mask'] for a in batch])
    
    return batch_input, batch_target, batch_target_mask

if __name__ == '__main__':
    """
    Data loading
    """
    with open(root_folder+"dataset/headline_generation_dataset_processed.json", "r") as f:
        d_released = json.load(f)
    with open(root_folder+"dataset/headline_generation_vocabulary.txt", "r",encoding='utf8') as f:
        vocabulary = f.read().split("\n")
    
    w2i = {w: i for i, w in enumerate(vocabulary)} # Word to index
    i2w = {i: w for i, w in enumerate(vocabulary)} # Index to word
    unkI, padI, startI = w2i['UNK'], w2i['PAD'], w2i['<START>']

    vocab_size = len(vocabulary)
    input_length = len(d_released[0]['numerized']) 
    d_train = [d for d in d_released if d['cut'] == 'training']
    d_valid = [d for d in d_released if d['cut'] == 'validation']

    """
    Define hyperparameters
    """

    hidden_size = 800   # previously 512, 1024 (better but too large)
    num_layers = 1      # previously 1 (when 512 unit per layer)
    lr = 1e-4           # <tune this> previously 1e-4
    dropout = 0.       # <tune this> previously .2
    weight_decay = 0.02 # <tune this> by default 0.01
    gpu_list = [5, 6, 7, 8, 9]
    optimizer_class = optim.AdamW
    epochs = 25         
    batch_size = 512    # initially 128
    resume_training = False
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    def loss_fn(pred, target, mask):
        pred = pred.permute(0, 2, 1)  # put the class probabilities in the middle
        loss_tensor = criterion(pred, target)
        loss_masked = loss_tensor * mask # your_code
        loss_per_sample = loss_masked.mean() # your_code
        return loss_per_sample

    batch_to_torch = lambda b_in,b_target,b_mask: (th.Tensor(b_in).long(),
                                                th.Tensor(b_target).long(), 
                                                th.Tensor(b_mask).float())

    model_id = 'test1'
    os.makedirs(root_folder+'models/part1/',exist_ok=True)

    device = th.device(f"cuda:{gpu_list[0]}" if th.cuda.is_available() else "cpu")
    print(device)
    list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]
    
    """
    Trainer code
    """
    losses = []
    accuracies = []

    """ START of model loading """
    if resume_training:
        ckpt_path = root_folder+'models/part1/'+f"model_{model_id}.pt"
        save_dict = th.load(ckpt_path)
        model = nn.DataParallel(LanguageModel(**save_dict['kwargs']), device_ids=gpu_list)
        model.load_state_dict(save_dict['model_state_dict'])
        print(f"===\nLoaded model checkpoint from: {ckpt_path}\n===")
    else:
        model = LanguageModel(vocab_size=vocab_size, rnn_size=hidden_size, num_layers=num_layers, dropout=dropout)
        model = nn.DataParallel(model, device_ids=gpu_list)
    model.cuda(device)
    """ END of model loading """
    print(f"Model is on: {gpu_list}")
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Training for {epochs} epoch, batch size {batch_size}, lr={lr}, weight_decay={weight_decay}")
    print(f"network config: {hidden_size} by {num_layers} layers.")
    for epoch in range(epochs):
        indices = np.random.permutation(range(len(d_train)))
        t = range((len(d_train)//batch_size)+1)
        for i in t:
            # Here is how you obtain a batch:
            batch = build_batch(d_train, indices[i*batch_size:(i+1)*batch_size])
            (batch_input, batch_target, batch_target_mask) = batch_to_torch(*batch)
            (batch_input, batch_target, batch_target_mask) = list_to_device((batch_input, batch_target, batch_target_mask))
            
            prediction = model(batch_input)
            loss = loss_fn(prediction, batch_target, batch_target_mask)
            losses.append(loss.item())
            accuracy = (th.eq(prediction.argmax(dim=2,keepdim=False),batch_target).float()*batch_target_mask).sum()/batch_target_mask.sum()
            accuracies.append(accuracy.item())
            
            """ YOUR CODE """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i == (len(d_train)//batch_size) // 2:
            #     print(f"Epoch: {epoch} Iteration: 50% Loss: {np.mean(losses[-10:])} Accuracy: {np.mean(accuracies[-10:])}")
        print(f"Epoch: {epoch} Iteration: 100% Loss: {np.mean(losses[-10:])} Accuracy: {np.mean(accuracies[-10:])}")

        # save your latest model
        save_dict = dict(
            kwargs = dict(
                vocab_size=vocab_size,
                rnn_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ),
            model_state_dict = model.state_dict(),
            notes = "",
            optimizer_class = optimizer_class,
            lr = lr,
            epochs = epochs,
            batch_size = batch_size,
        )
        th.save(save_dict,root_folder+f'models/part1/model_{model_id}.pt')
