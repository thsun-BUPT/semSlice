
# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/31 
@Author ：Witty
@File ：pso.py
"""


import sys
import os
import time
import json
 
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
        pass
 
    def close(self):
        self.log.close()
 

script_name = os.path.splitext(os.path.basename(__file__))[0]
start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_dir = f"./Records/semanslice/SS/fitTASK/{script_name}_{start_time}/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  
log_filename = os.path.join(log_dir, f"{script_name}_{start_time}.log")


sys.stdout = Logger(log_filename)


import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

from sem_slice_SS_15TASK import main_compute

class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high,target_snr):

        self.dimension = dimension 
        self.time = time  
        self.gen = 0 
        self.size = size  
        self.bound = []  
        self.bound.append(low) 
        self.bound.append(up) 
        
        self.v_low = v_low 
        self.v_high = v_high
       
        self.x = np.random.uniform(0.01, 0.9, (self.size, self.dimension))
        self.v = np.zeros((self.size, self.dimension))  
     
        self.p_best = np.random.uniform(0.01, 0.9, (self.size, self.dimension))

        self.g_best = np.random.uniform(0.01, 0.9, self.dimension)
        self.p_best_fitness = np.zeros((self.size)) 
        self.g_best_fitness = 0
        self.best_eachslice=[]
        self.target_snr = target_snr  
        self.d = 3000  
        self.n0 = 10 ** (-114.45 / 10) * 1e-3  


        for i in range(self.size): 
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
           
                self.v[i][j] = random.uniform(self.v_low, self.v_high) 
            self.p_best[i] = self.x[i] 
          
               
    def fitness(self, x): 
        """
        个体适应值计算
        """
        snr_linear = 10 ** (self.target_snr / 10)
        for i in range(3):  
            p = x[i]
            b = x[i + 3]

            if b > 0:
                required_power = snr_linear * b * 1e6 * (self.d ** 2) * self.n0
                if required_power <= P_total / 3:  
                    x[i] = required_power
                else:
                    x[i] = P_total / 3  

        total_power = sum(x[:3])
        total_bandwidth = sum(x[3:])
        
        if total_power > P_total:
            for i in range(3):
                x[i] = x[i] * P_total / total_power

        if total_bandwidth > B_total:
            for i in range(3):
                x[i + 3] = x[i + 3] * B_total / total_bandwidth
        
        sum_ssimilarity,each,results= main_compute(x)
        
        return sum_ssimilarity,each,results
 
    def update(self, size, c1, c2, w):
        for i in range(size):

            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])

            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high
 
            self.x[i] = self.x[i] + self.v[i]
     
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]

 
            current_fitness,each_slice,ep_results = self.fitness(self.x[i])

            data_results[(self.size*self.gen)+i] = ep_results
            

  
            if current_fitness > self.p_best_fitness[i]:
                self.p_best[i] = self.x[i]
                self.p_best_fitness[i] = current_fitness

            if current_fitness > self.g_best_fitness:
                self.g_best = self.x[i]
                self.g_best_fitness = current_fitness
                self.best_eachslice = each_slice

    def pso(self):
        best = []
        X1 = []
        X2 = []
        
        self.final_best = np.array([0.1,0.2,0.3,0.1,1.5,0.1])

        with tqdm(total=self.size * self.time) as percent:
            for gen in range(self.time):
                self.gen = gen
                c1 = 1.5 + np.sin(math.pi/2*(1-(2*gen/self.time))) 
                c2 = 1.5 + np.sin(math.pi/2*((2*gen/self.time)-1))
                w = 1.6 - 1.2*gen/self.time
                print('\n --------当前为第{}次迭代------'.format(gen))
                self.update(self.size, c1 ,c2, w)
                percent.update(self.size)

                self.final_best = self.g_best.copy()
                print('\n')
                print('\n 第{}次迭代,当前最佳位置：{}'.format(gen,self.final_best))
                temp = self.g_best_fitness
                print('\n 第{}次迭代,当前的最佳适应度：{}'.format(gen,temp))
                print('\n 第{}次迭代,当前最佳的每个切片的适应度：{}'.format(gen,self.best_eachslice))
                print('\n --------第{}次迭代结束------\n'.format(gen))
                best.append(temp)
                X1.append(self.final_best[0])
                X2.append(self.final_best[1])
                
            
        
        print("\n ----------------所有迭代结束---------------------------")
        print('\n 最终的最佳分配：{}'.format(self.final_best))
        print('\n 最终的最佳ssimilarity：{}'.format(self.g_best_fitness))
        print('\n 最终最佳的每个切片的ssimilarity：{}'.format(self.best_eachslice))


        t = [i for i in range(self.time)]
        plt.figure()
        plt.grid(ls='--')
        plt.plot(t, best, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  
        plt.ylabel(u"最大值")  
        plt.title(u"迭代过程")  
        fig1_path = os.path.join(log_dir, 'PSO_最大值.png')
        plt.savefig(fig1_path)

        plt.figure()
        plt.grid(axis='both')
        plt.plot(t, X1, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数") 
        plt.ylabel(u"X1")
        plt.title(u"X1识别曲线") 
        fig2_path = os.path.join(log_dir, 'PSO_x1识别曲线.png')
        plt.savefig(fig2_path)
 
        plt.figure()
        plt.grid(axis='both')
        plt.plot(t, X2, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  
        plt.ylabel(u"X2")  
        plt.title(u"X2识别曲线") 
        fig3_path = os.path.join(log_dir, 'PSO_x2识别曲线.png')
        plt.savefig(fig3_path)

 
if __name__ == '__main__':

    time = 10 
    size = 5
    dimension = 6
    v_low = -0.1
    v_high = 0.1
    low = [0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001]

    up = [0.999, 0.999, 0.999, 1.999, 1.999, 1.999]
    P_total=1 
    B_total=2 
    target_SNR = 6 

   
    data_results = [[{} for _ in range(15)] for _ in range(time*size)]

    pso = PSO(dimension, time, size, low, up, v_low, v_high,target_SNR)
    pso.pso()


    results_dir = f"./Records/semanslice/SS/fitTASK/{script_name}_{start_time}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  
    results_filename = os.path.join(results_dir, f"{script_name}_{start_time}.json")

    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(data_results, f, ensure_ascii=False, indent=4)
    print(f"\n Data has been saved to {results_filename}")
   







