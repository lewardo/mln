# Machine Learning Network
## a simple implementation of a simple mlp cli

---
>## ___WARNING:___
>***This code is broken***, I found it in the depths of my old laptop. I currently don't have the time to fix it up, but the next commit will be a brand new, shiny working version!  
>Points included for next version:
> + proper documentation
> + new commands for cleaner cli
> + simpler useage
>
>I can't remember when this last worked, I think it was my first attempt at an mlp. [libml](https://www.github.com/lewardo/libml) is a more fully fledged (albiet not clified) cpp ml library under current development, and will probably get more attention than this project, same goes for [mlp-nn](https://www.github.com/lewardo/mlp-nn) (basically a mlp-only version of libml, although as of writing, is the only working one)
---
### **flags**

| flag     | meaning                          |
|----------|----------------------------------|
| -t/-p    | mode selection (train/predict)   |
| -i [str] | input file path                  |
| -o [str] | output file path                 |
| -e [int] | max num of epochs                |
| -n [str] | network topology definition path |

### **Training Input File Format:**
```bash
\[ni\] \[nl\]  
\[L1\] \[L2\] \[L3\] ...  
\[i1\] \[i2\] \[i3\] ... \[o1\] \[o2\] \[o3\] ...  
\[i1\] \[i2\] \[i3\] ... \[o1\] \[o2\] \[o3\] ...  
\[i1\] \[i2\] \[i3\] ... \[o1\] \[o2\] \[o3\] ...  
...  
```
where `ni` = number of training data input-output pairs
	`nl` = number of layers in nn
	`L1`, `L2`, `L3` ... = number of neurons per layer
	`i1`, `i2`, `i3` ... = inputs associated with
	`o1`, `o2`, `o3` ... = outputs given here

---

### **Predicting Input File Format:**
```bash
\[ni\]  
\[i1\] \[i2\] \[i3\] ...  
\[i1\] \[i2\] \[i3\] ...  
\[i1\] \[i2\] \[i3\] ...  
...
```
where `ni` = number of inputs
	`i1`, `i2`, `i3` ... = inputs to be predicted

---

### **Prediction Output File:**
```bash
\[o1\] \[o2\] \[o3\] ...  
\[o1\] \[o2\] \[o3\] ...  
\[o1\] \[o2\] \[o3\] ...  
...
```
where `o1`, `o2`, `o3` = outputs corresponding to respective input from input file

---

### **example useage:**
```bash
> g++ main.cpp lib/mlp.cpp -o main -O3 -std=c++11
> ./main -t -i files/train.txt -o files/nn.txt -e 100000
input file files/train.txt, mode t
nn trained 100000 times, training error: 0.000027
> ./a.out -p -i files/predict.txt -o files/out.txt -n files/nn.txt
data input file files/predict.txt, mode p
output file files/out.txt
```
---
</> with ❤️ by lewardo 2021