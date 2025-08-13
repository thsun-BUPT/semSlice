# SemSlice: Semantic Communication-Oriented Network Slicing Framework  
Yu Zhou, Lei Feng, Member, IEEE, Fanqin Zhou, Member, IEEE, Yao Sun, Senior Member, IEEE, Tianhang Sun, Celimuge Wu, Senior Member, IEEE, Wenjing Li, Member, IEEE, Zehui Xiong, Senior Member, IEEE  


This is the implementation of SemSlice: Semantic Communication-Oriented Network Slicing Framework  


## Requirements  
See the `requirements.txt` for the required python packages and run the following command to install them:  
```bash  
pip install -r requirements.txt  
```  


## Run the Code  
We have trained the DeepSC model, you can get it by loading the link:  
Link：https://pan.baidu.com/s/1gGNaAYpBIXfeD9FHFKMUvw  
Password：1234  


### To implement fitSNR task:  
- SemSlice:  
  ```bash  
  python pso_semanslice_SS_fitSNR.py  
  ```  
- NetSlice:  
  ```bash  
  python pso_netslice_SS_fitSNR.py  
  ```  
- Random:  
  ```bash  
  python no_slice_random_SS.py  
  ```  


### To implement fitTASK task:  
#### SemSlice:  
- 5 Tasks:  
  ```bash  
  python pso_semanslice_SS_fit5TASK.py  
  ```  
- 15 Tasks:  
  ```bash  
  python pso_semanslice_SS_fit15TASK.py  
  ```  

#### NetSlice:  
- 5 Tasks:  
  ```bash  
  python pso_netslice_random_SS_fit5TASK.py  
  ```  
- 15 Tasks:  
  ```bash  
  python pso_netslice_random_SS_fit15TASK.py  
  ```  

#### Random:  
- 5 Tasks:  
  ```bash  
  python no_slice_random_KPI_fit5TASK.py  
  ```  
- 15 Tasks:  
  ```bash  
  python no_slice_random_KPI_fit15TASK.py  
  ```  


### To implement Emergency Task:  
- SemSlice:  
  ```bash  
  python emergency_process_semslice.py  
  ```  
- NetSlice:  
  ```bash  
  python emergency_process_netslice.py  
  ```  
- Random:  
  ```bash  
  python emergency_process_noslice.py  
  ```


## Acknowledgements  
This project builds on the foundational work presented in the paper:  

- H. Xie, Z. Qin, G. Y. Li and B. -H. Juang, "Deep Learning Enabled Semantic Communication Systems," in *IEEE Transactions on Signal Processing*, vol. 69, pp. 2663-2675, 2021, doi: [10.1109/TSP.2021.3071210](https://doi.org/10.1109/TSP.2021.3071210).  

The link of this work：[https://github.com/13274086/DeepSC](https://github.com/13274086/DeepSC)