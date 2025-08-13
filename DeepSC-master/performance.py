

# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser()

parser.add_argument('--vocab-file', default='europarl/vocab_en.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/1201/deepsc-AWGN-enorigin_layer3', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=20, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=3, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type = int)
parser.add_argument('--bert-config-path', default='./cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='./cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='./cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")





def performance(args, SNR, net):

    bleu_score_1gram = BleuScore(0.25, 0.25, 0.25, 0.25)

    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
       
                    target = sents

                    out,_ = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)
            print("TX_word:",len(Tx_word))
            print("RX_word:",len(Rx_word))
           
            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
       
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) 
    
                print("bleu_score:", np.mean(bleu_score))

            bleu_score = np.array(bleu_score)

            bleu_score = np.mean(bleu_score, axis=1)

            score.append(bleu_score)
            print("score:",score)


    score1 = np.mean(np.array(score), axis=0)

    return score1

if __name__ == '__main__':
    args = parser.parse_args()

    SNR=[6]
    args.vocab_file = './' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1]) 
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)

    print('data load!')

    bleu_score  = performance(args, SNR, deepsc)
 
    print("bleu_score is ",bleu_score)

    print("vocab is:",args.vocab_file,"checkpoint is:",args.checkpoint_path)


                             