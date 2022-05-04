import json
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformer import Transformer
from transformer_utils import set_device
import sentencepiece as spm
from tqdm import tqdm
import os

root_folder = "./"

def build_batch(dataset, batch_size):
    indices = list(np.random.randint(0, len(dataset), size=batch_size))
    
    batch = [dataset[i] for i in indices]
    batch_input = np.array([a['input'] for a in batch])
    batch_input_mask = np.array([a['input_mask'] for a in batch])
    batch_output = np.array([a['output'] for a in batch])
    batch_output_mask = np.array([a['output_mask'] for a in batch])
    
    return batch_input, batch_input_mask, batch_output, batch_output_mask

class TransformerTrainer(nn.Module):
    def __init__(
        self, vocab_size, d_model, input_length, output_length, 
        n_layers, d_filter, dropout=0, learning_rate=1e-3, 
        ckpt=None, multi_gpu=False, gpu_list=None):
        super().__init__()
        # if multi_gpu and gpu_list != None:
        #     self.model = nn.DataParallel(self.model, device_ids=gpu_list)
        if ckpt == None:
            self.model = Transformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_filter=d_filter)
        else:
            save_dict = th.load(ckpt)
            self.model = Transformer(**save_dict['kwargs'])
            self.model.load_state_dict(save_dict['model_state_dict'])
            print(f"Loaded checkpoint from {ckpt}")

        # Summarization loss
        criterion = nn.CrossEntropyLoss(reduce='none')
        self.loss_fn = lambda pred,target,mask: (criterion(pred.permute(0,2,1),target)*mask).sum()/mask.sum()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self,batch,optimize=True):
        pred_logits = self.model(**batch)
        target,mask = batch['target_sequence'],batch['decoder_mask']
        loss = self.loss_fn(pred_logits,target,mask)
        accuracy = (th.eq(pred_logits.argmax(dim=2,keepdim=False),target).float()*mask).sum()/mask.sum()
        
        if optimize:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
        return loss, accuracy


def pad_sequence(numerized, pad_index, to_length):
    pad = numerized[:to_length]
    padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask

if __name__ == "__main__":

    gpu_list = [9, ] # Use whatever available on the cluster
    device = th.device(f"cuda:{gpu_list[0]}")
    print(f"Using device cuda: {gpu_list}")
    set_device(device)
    list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]

    # Load the word piece model that will be used to tokenize the texts into
    # word pieces with a vocabulary size of 10000
    sp = spm.SentencePieceProcessor()
    sp.Load(root_folder+"dataset/wp_vocab10000.model")

    vocab = [line.split('\t')[0] for line in open(root_folder+"dataset/wp_vocab10000.vocab", "r")]
    pad_index = vocab.index('#')

    with open(root_folder+"dataset/summarization_dataset_preprocessed.json", "r") as f:
        dataset = json.load(f)

    # We load the dataset, and split it into 2 sub-datasets based on if they are training or validation.
    # Feel free to split this dataset aanother way, but remember, a validation set is important, to have an idea of 
    # the amount of overfitting that has occurred!

    d_train = [d for d in dataset if d['cut'] == 'training']
    d_valid = [d for d in dataset if d['cut'] == 'evaluation']

    print(f"Training set size = {len(d_train)}; Validation set size = {len(d_valid)}.")

    """
    Hyperparameters
    """
    # Dataset related parameters
    vocab_size = len(vocab)
    ilength = 400 # Length of the article
    olength  = 100 # Length of the summaries

    # Model related parameters, feel free to modify these.
    n_layers = 6
    d_model  = 160
    d_filter = 4*d_model
    batch_size = 64
    resume_training = True

    dropout = 0
    learning_rate = 1e-3
    model_id = 'test1'
    trainer = TransformerTrainer(
        vocab_size, d_model, ilength, olength, n_layers, d_filter, dropout, 
        ckpt=root_folder+'models/part2/'+f"model_{model_id}.pt",
        multi_gpu=True, gpu_list=gpu_list)
    os.makedirs(root_folder+'models/part2/',exist_ok=True)

    """
    Training section
    """

    trainer.to(device)
    trainer.model.to(device)

    trainer.model.train()
    losses,accuracies = [],[]

    iterations = 1e4 # started with 3e4 already
    # t = tqdm(range(int(iterations)+1))
    t = range(int(iterations)+1)
    for i in t:
        # Create a random mini-batch from the training dataset
        batch = build_batch(d_train, batch_size)
        # Build the feed-dict connecting placeholders and mini-batch
        batch_input, batch_input_mask, batch_output, batch_output_mask = [th.tensor(tensor) for tensor in batch]
        batch_input, batch_input_mask, batch_output, batch_output_mask \
                    = list_to_device([batch_input, batch_input_mask, batch_output, batch_output_mask])
        batch = {'source_sequence': batch_input, 'target_sequence': batch_output,
                'encoder_mask': batch_input_mask, 'decoder_mask': batch_output_mask}

        # Obtain the loss. Be careful when you use the train_op and not, as previously.
        train_loss, accuracy = trainer(batch)
        losses.append(train_loss.item()),accuracies.append(accuracy.item())
        # if i % 10 == 0:
        #     t.set_description(f"Iteration: {i} Loss: {np.mean(losses[-10:])} Accuracy: {np.mean(accuracies[-10:])}")
        if i % 100 == 0:
            print(f"Iteration: {i} Loss: {np.mean(losses[-10:])} Accuracy: {np.mean(accuracies[-10:])}")
            save_dict = dict(
                kwargs = dict(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers, 
                    d_filter=d_filter
                ),
                model_state_dict = trainer.model.state_dict(),
                notes = ""
            )
            th.save(save_dict, root_folder+f'models/part2/model_{model_id}.pt')