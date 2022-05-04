import numpy as np
import os
import json
import torch as th
from segtok import tokenizer
from scipy.special import softmax
from torch import nn
from utils import validate_to_array
from language_model import LanguageModel
from trainer_file import numerize_sequence, pad_sequence

root_folder = "./"
gpu_list = [8, 9]
device = th.device(f'cuda:{gpu_list[0]}' if th.cuda.is_available() else 'cpu')
list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]
batch_to_torch = lambda b_in,b_target,b_mask: (th.Tensor(b_in).long(),
                                                th.Tensor(b_target).long(), 
                                                th.Tensor(b_mask).float())

criterion = nn.CrossEntropyLoss(reduction='none')

def numerize_sequence(tokenized):
    return [w2i.get(w, unkI) for w in tokenized]
def pad_sequence(numerized, pad_index, to_length):
    pad = numerized[:to_length]
    padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask

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
    
def loss_fn(pred, target, mask):
    pred = pred.permute(0, 2, 1)  # put the class probabilities in the middle
    loss_tensor = criterion(pred, target)
    loss_masked = loss_tensor * mask # your_code
    loss_per_sample = loss_masked.mean() # your_code
    return loss_per_sample


def raw_sample_pred(headline, model):
    tokenized = tokenizer.word_tokenizer(headline.lower())

    numerized = numerize_sequence(tokenized)
    padded, mask = pad_sequence(numerized, padI, input_length)

    input_headline = th.Tensor(np.array([startI] + padded[:-1])[None, :]).long()
    pred_headline = model(input_headline)
    target_headline = th.Tensor(np.array(padded)[None, :]).long()
    mask = th.Tensor(mask).float()

    return pred_headline.cpu(), target_headline.cpu(), mask.cpu()

def generate_sentence(headline_starter, model):
    
    tokenized = tokenizer.word_tokenizer(headline_starter.lower())
    current_build = [startI] + numerize_sequence(tokenized)
    while len(current_build) < input_length:
        current_padded, _m = pad_sequence(current_build, padI, input_length)

        # Obtain the logits for the current padded sequence
        np_input = np.array(current_padded, dtype=np.float64)[None, :]
        logits = model(th.Tensor(np_input).long())
        logits_np = logits.detach().cpu().numpy()

        # Obtain the logits for the last non-pad inputs
        last_logits = logits_np[:, len(current_build)-1, :]

        # Find the highest scoring words in the last_logits
        # array, or sample from the softmax.
        # The np.argmax function may be useful for first option,
        # sp.special.softmax and np.random.choice may be useful for second option.
        # Append this word to our current build
        current_build.append( np.random.choice(range(0,10000), p=softmax(last_logits[0])) )

    produced_sentence = ' '.join([i2w[c] for c in current_build])
    return produced_sentence

if __name__ == "__main__":
    """ Load in the validation dataset """
    with open(root_folder+"dataset/headline_generation_dataset_processed.json", "r") as f:
        d_released = json.load(f)
    with open(root_folder+"dataset/headline_generation_vocabulary.txt", "r",encoding='utf8') as f:
        vocabulary = f.read().split("\n")
    w2i = {w: i for i, w in enumerate(vocabulary)} # Word to index
    i2w = {i: w for i, w in enumerate(vocabulary)} # Index to word
    unkI, padI, startI = w2i['UNK'], w2i['PAD'], w2i['<START>']

    vocab_size = len(vocabulary)
    input_length = len(d_released[0]['numerized']) 
    d_valid = [d for d in d_released if d['cut'] == 'validation']
    
    """ Evalutaion of loss """

    model_id = "test1"
    save_dict = th.load(root_folder + 'models/part1/'+f"model_{model_id}.pt",)    
    model = nn.DataParallel(LanguageModel(**save_dict['kwargs']), device_ids=gpu_list)
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda(device=device)
    model.eval()

    batch = build_batch(d_valid, range(len(d_valid)))
    (batch_input, batch_target, batch_target_mask) = batch_to_torch(*batch)
    (batch_input, batch_target, batch_target_mask) = list_to_device((batch_input, batch_target, batch_target_mask))
    prediction = model(batch_input.long())
    loss = loss_fn(prediction, batch_target, batch_target_mask)
    print("Evaluation set loss:", loss.item())

    os.makedirs(root_folder+"best_models",exist_ok=True)
    best_model_file = root_folder+"best_models/part1_best_model_multiGPU.pt"
    th.save(save_dict,best_model_file)
    # model.cpu()

    """ Evaluation of likelihood of data """
    
    model.eval()

    headline1 = "Apple to release new iPhone in July"
    headline2 = "Apple and Samsung resolve all lawsuits"

    headlines = [headline1.lower(), headline2.lower()] # Our LSTM is trained on lower-cased headlines
    for headline in headlines:
        pred_headline,target_headline,mask = raw_sample_pred(headline, model)
        loss = loss_fn(pred_headline.to(device), target_headline.to(device), mask.to(device))       
        print("----------------------------------------")
        print("Headline:", headline)
        print("Loss of the headline:", loss)
    with th.no_grad():
        validate_to_array(raw_sample_pred,zip(headlines,[model]*2),'raw_sample_pred',root_folder,multi=True)
    
    """ Generate headlines """

    model.eval()
    headline_starters = ["apple has released", "google has released", "amazon", "tesla to", "facebook is now meta", "youtube changes its user policy"]
    for headline_starter in headline_starters:
        # print("===================")
        # print("Generating headline starting with: "+headline_starter)

        produced_sentence = generate_sentence(headline_starter, model)
        print(produced_sentence)
    with th.no_grad():
        validate_to_array(generate_sentence,zip(headline_starters,[model]*len(headline_starters)),"generate_sentence",root_folder,multi=True)