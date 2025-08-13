import os
import sys
import time


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')  
 
    def write(self, message):
   
        if "searching:" not in message.strip():
            if message.strip():  
                self.terminal.write(message)
                self.log.write(message)
  
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
     
 
    def close(self):
        self.log.close()

script_name = os.path.splitext(os.path.basename(__file__))[0]
start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_dir = f'./Records/noslice/KPI/fitTASK/{script_name}_{start_time}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  
log_filename = os.path.join(log_dir, f"{script_name}_{start_time}.log")


sys.stdout = Logger(log_filename)

import random
import json
import torch
import argparse
import pickle
import numpy as np

from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from tensorflow import keras
from bert4keras.backend import keras
from bert4keras.bert import build_bert_model 
from bert4keras.tokenizer import Tokenizer 
from w3lib.html import remove_tags

from utils import *
from models.transceiver import DeepSC


import os
import tensorflow as tf
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
logging.getLogger("bert4keras").setLevel(logging.ERROR)




parser = argparse.ArgumentParser()

'''CHANGE'''
parser.add_argument('--checkpoint-path_1', default='checkpoints/1201/deepsc-AWGN-enorigin_layer3', type=str)
parser.add_argument('--checkpoint-path_2', default='checkpoints/1201/deepsc-AWGN-en90%_layer3', type=str)
parser.add_argument('--checkpoint-path_3', default='checkpoints/1201/deepsc-AWGN-en80%_layer3', type=str)

parser.add_argument('--task_1_vocab', default='./europarl/vocab_en.json', type=str)
parser.add_argument('--task_2_vocab', default='./europarl/vocab_en.json', type=str)
parser.add_argument('--task_3_vocab', default='./europarl/vocab_en90%.json', type=str)
parser.add_argument('--task_4_vocab', default='./europarl/vocab_en90%.json', type=str)
parser.add_argument('--task_5_vocab', default='./europarl/vocab_en80%.json', type=str)


parser.add_argument('--task_1', default='./europarl/test_data_en.pkl', type=str)
parser.add_argument('--task_2', default='./europarl/test_data_en.pkl', type=str)
parser.add_argument('--task_3', default='./europarl/test_data-en90%.pkl', type=str)
parser.add_argument('--task_4', default='./europarl/test_data-en90%.pkl', type=str)
parser.add_argument('--task_5', default='./europarl/test_data-en80%.pkl', type=str)



parser.add_argument('--vocab-file', default='europarl/vocab_en.json', type=str)
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

parser.add_argument('--sim_threshold',default= 0.6,type = float)
parser.add_argument('--delay_threshold',default= 0.13,type = float) 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tasks_num = 5
slices_num = 3


      




args = parser.parse_args()
args.vocab_file = './' + args.vocab_file 
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]



""" 辅助函数"""
class Similarity():
    def __init__(self, config_path, checkpoint_path, dict_path):
        self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
        self.model = keras.Model(inputs=self.model1.input,
                                 outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
        # build tokenizer
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def compute_similarity(self, real, predicted):
        token_ids1, segment_ids1 = [], []
        token_ids2, segment_ids2 = [], []
        score = []

        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            ids1, sids1 = self.tokenizer.encode(sent1)
            ids2, sids2 = self.tokenizer.encode(sent2)

            token_ids1.append(ids1)
            token_ids2.append(ids2)
            segment_ids1.append(sids1)
            segment_ids2.append(sids2)

        token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
        token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')

        segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
        segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')

        vector1 = self.model.predict([token_ids1, segment_ids1])
        vector2 = self.model.predict([token_ids2, segment_ids2])

        vector1 = np.sum(vector1, axis=1)
        vector2 = np.sum(vector2, axis=1)

        vector1 = normalize(vector1, axis=0, norm='max')
        vector2 = normalize(vector2, axis=0, norm='max')

        dot = np.diag(np.matmul(vector1, vector2.T))  
        a = np.diag(np.matmul(vector1, vector1.T))  
        b = np.diag(np.matmul(vector2, vector2.T))

        a = np.sqrt(a)
        b = np.sqrt(b)

        output = dot / (a * b)
        score = output.tolist()

        return score
    

class EurDataset(Dataset):
    def __init__(self, task_idx, args):
        path = getattr(args, f'task_{task_idx + 1}', None)
        if path is None:
            raise ValueError(f"Path for task {task_idx + 1} not found.")
        try:
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load data from {path}: {e}")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def get_latest_model_path(encoder_dir):
    """
    获取指定目录中最新的模型路径。
    参数:
        encoder_dir (str): 包含模型文件的目录路径。
    返回:
        str: 最新的模型文件路径。
    """
    model_paths = []
    for fn in os.listdir(encoder_dir):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  
        model_paths.append((os.path.join(encoder_dir, fn), idx))

    if not model_paths:
        raise FileNotFoundError(f"No .pth files found in {encoder_dir}")
    
    model_paths.sort(key=lambda x: x[1])  

    latest_model_path, _ = model_paths[-1]
    return latest_model_path

def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   

    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  

    return  torch.from_numpy(sents)



def compute_SNR(P,B):
    N0=-114.45 
  
    N0=10**(N0/10)*(10**(-3))

    d=3000 
   
    SNR = P/(B*(10**6)*(d**2)*N0)
    SNR = 10*np.log10(SNR)
 
    return SNR


def transmit_delay(k,L,B,SNR):

    SNR = 10**(SNR/10)
    C= B*(10**6) * math.log2(1+SNR)
    t=k*L/C

    return t



'''  主要函数  '''

def tasks(args): 
    """
    任务分配
    """
 
    
    task_list=[[] for _ in range(tasks_num)] 
    slices_list=[[] for _ in range(slices_num)] 
    task_data_set={}
    deepsc_models={}

    while True:
        encoder_idxs = [random.randint(1, 3) for _ in range(tasks_num)]
        if 1 in encoder_idxs and 2 in encoder_idxs and 3 in encoder_idxs:
            break

    Datasets = {i: EurDataset(i, args) for i in range(tasks_num)} 

    for slice_idx in range(slices_num): 
        checkpoint_path = getattr(args, f'checkpoint_path_{slice_idx + 1}', None)
        if checkpoint_path is None:
            raise ValueError(f"Checkpoint path for slice {slice_idx + 1} not found.")
        
        """ define optimizer and loss function """
        deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                num_vocab, num_vocab, args.d_model, args.num_heads,
                args.dff, 0.1).to(device)
        
        """获取最新的模型路径并加载权重 """
        latest_model_path = get_latest_model_path(checkpoint_path)
        checkpoint = torch.load(latest_model_path)
        deepsc.load_state_dict(checkpoint)
        
        deepsc_models[slice_idx] = deepsc
        print('\n model load! slice {}'.format(slice_idx+1))

    for i in range(tasks_num):
        task_list[i].append(i+1) 
        encoder_idx = encoder_idxs[i] 
        if encoder_idx == 1:
            slices_list[0].append(i+1) 
        elif encoder_idx == 2:
            slices_list[1].append(i+1)
        elif encoder_idx == 3:
            slices_list[2].append(i+1)
        task_list[i].append(encoder_idxs[i])

      
        test_iterator = DataLoader(Datasets[i], batch_size=args.batch_size, num_workers=0,
                                   pin_memory=True, collate_fn=collate_data)
        task_data_set[i] = test_iterator


    print("\n Task List is: ",task_list) 
    print("\n Slices List is:",slices_list) 
    print('\n Task Data Set loading')
    print('\n ---- 加载完成--- \n')
    return task_list,slices_list,task_data_set,deepsc_models



def performance(args, SNR, Encoder_idx,Task_idx):
    similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)



    net = deepsc_models[Encoder_idx]

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []

    net.eval()
    with torch.no_grad():
        for epoch in range(1):
            Tx_word = []
            Rx_word = []

            for snr in range(len(SNR)):
                word = []
                target_word = []
                noise_std = SNR_to_noise(SNR[snr])

                for sents in task_data_set[Task_idx]:

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
       
           
    
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
          
   

                sim_score.append(similarity.compute_similarity(sent1, sent2)) 
                print("\n sim_score:", np.mean(sim_score))
       
            sim_score = np.array(sim_score)

            sim_score = np.mean(sim_score, axis=1)

            score.append(sim_score)
            print("\n score:",score)


    score1 = np.mean(np.array(score), axis=0)

    return score1, _




def compute_performance(P,B,k,Encoder_idx,Task_idx):
  
    L=30
    SNR=[]
    SNR.append(compute_SNR(P,B))


    sim_score, _  = performance(args, SNR, Encoder_idx,Task_idx)

    delay_transmit = transmit_delay(k, L, B, SNR[0])

    total_delay = 0 + delay_transmit

    print("\n  ---- ")
    print("\n P iS ",P)
    print("\n B iS ",B)
    print("\n SNR iS ",SNR[0])
    print("\n total_delay （no process delay） is ",total_delay,"s")

    print("\n delay_transmit is ", delay_transmit*(10**3),"ms") 
    print("\n sim_score is ",sim_score)
    print("\n -------------\n")

    
    return sim_score[0],total_delay,delay_transmit*(10**3)








if __name__ == '__main__':



    ssimilarity = None
    total_ssimilarity=[]
    avg_ssimilarity=[]
    s_se = None
    total_S_SE=[]
    avg_S_SE=[]


    args = parser.parse_args()

    epoch = 5
    P_total = 1 
    B_total = 2 
    k = 10
    
    data_results = [[{} for _ in range(tasks_num)] for _ in range(epoch)]

    print('-------- 无切片实验-------- \n')
    for t in range(epoch):
        ssimilarity_list=[]
        ssimilarity_mean_list=[0.0] * slices_num 

        S_SE_list=[]
        S_SE_mean_list=[0.0] * slices_num

        
        print('--------------Epoch {}/{}  Beginning! ------------- \n'.format(t+1,epoch))

        tasks_list,slices_list,task_data_set,deepsc_models = tasks(args)
    
        for i in range(tasks_num):
            print('-------------------- \n')


            P=0.0947
            B=0.667 


            Task_idx = i 
            Encoder_idx = tasks_list[i][1] - 1 
            print("切片号：{}，任务号：{}, 进度 {}/{} \n".format(Encoder_idx+1,Task_idx+1,i+1,tasks_num))
            
            '''CHANGE'''
            sim_score,total_delay,delay_transmit = compute_performance(P,B,k,Encoder_idx,Task_idx)
            ssimilarity = sim_score 
            if 0<= i <=2:
                if ssimilarity < args.sim_threshold: 
                    ssimilarity=0
            if 3<= i <=4:
                if delay_transmit > args.delay_threshold: 
                    ssimilarity=0

            s_se_score = sim_score/ k
            s_se = ssimilarity/k 

            print("\n s_se_score is ",s_se_score)
            print("\n -------------\n")
            print("****SS Is:{} ******\n".format(ssimilarity))
            print("****Semantic Spectral efficiency Is:{} I/L ******\n".format(s_se))

            ssimilarity_list.append(ssimilarity)
            ssimilarity_mean_list[Encoder_idx] += ssimilarity  
  
            S_SE_list.append(s_se)
            S_SE_mean_list[Encoder_idx] += s_se
            data_results[t][i] = {
                "Task_id": Task_idx, 
                "Slice_id": Encoder_idx, 
                "Similarity": sim_score,
                "Delay": delay_transmit, 
                "S-SE": s_se_score
            }

        for j in range(slices_num):
            ssimilarity_mean_list[j] /= sum(1 for task_result in data_results[t] if task_result["Slice_id"] == j )
            S_SE_mean_list[j] /= sum(1 for task_result in data_results[t] if task_result["Slice_id"] == j )



        total_ssimilarity.append(ssimilarity_mean_list)
        total_S_SE.append(S_SE_mean_list)
        

        print("\n 当前的SNR为: ",compute_SNR(P,B))
        print("\n 各epoch各切片的similarity是:{}".format(total_ssimilarity))
        print("\n ssimilarity_list is(各切片各任务的similarity为):",ssimilarity_list)
        print("\n ssimilarity_mean_list is(每个切片的similarity为):",ssimilarity_mean_list)
        print("\n S_SE_list is(各切片各任务的S-SE为):",S_SE_list)
        print("\n S_SE_mean_list is(每个切片的S-SE为):",S_SE_mean_list)
        print('\n --------------Epoch {}/{}  End! -------------'.format(t+1,epoch))

    print("\n -------实验结束------")
    print("\n 当前的SNR为: ",compute_SNR(P,B))
    print("\n 各epoch各切片的ssimilarity是:{}".format(total_ssimilarity))
    print("\n 各epoch各切片的S-SE是:{}".format(total_S_SE))
    average_sim_per_slice = [sum(row[j] for row in total_ssimilarity) / epoch for j in range(slices_num)]
    average_S_SE_per_slice = [sum(row[j] for row in total_S_SE) / epoch for j in range(slices_num)]
    
    print("\n {}轮后,各切片的平均ssimilarity是:{}".format(epoch,average_sim_per_slice))
    print("\n {}轮后,各切片的平均S-SE是:{}".format(epoch,average_S_SE_per_slice))
    
 
    results_dir = f"./Records/noslice/KPI/fitTASK/{script_name}_{start_time}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  
    results_filename = os.path.join(results_dir, f"{script_name}_{start_time}.json")
  
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(data_results, f, ensure_ascii=False, indent=4)
    print(f"\n Data has been saved to {results_filename}")
    
            
        
        
    
    
            
        
        