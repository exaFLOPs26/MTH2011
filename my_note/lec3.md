> 수학적인 사고는 연역적으로 assumption을 바탕으로 풀어나가는 것 이를 통해 세상을 조금 더 설명력있게 but 현실과는 조금 떨어질 수도...

## MLE --> LS (Regression)  
> Maximizing the following function  
> <img width="271" alt="image" src="https://github.com/user-attachments/assets/8120b87f-5e0f-4880-b603-b81c55f6f525">
### Steps
1. Assume our hypothesis as linear model w\ noise  
    <img width="147" alt="image" src="https://github.com/user-attachments/assets/0f3cea65-f5bb-4a5f-8b4a-f1f76fe64218">  

2. Assumptions on noise  
    <img width="431" alt="image" src="https://github.com/user-attachments/assets/b1375f35-c4cc-4f6e-924f-ad92da1b2ff3">
3. Use Gaussian distribution for our model and parameter  
    <img width="431" alt="image" src="https://github.com/user-attachments/assets/1a69f09a-38d2-4714-84d0-9467b330545e">

4. By defining (log) likelihood, solving least squares is solving a maximum likelihood problem for a particular probabilistic model. 
    <img width="483" alt="image" src="https://github.com/user-attachments/assets/180fe6f9-4558-466b-b48b-e82b2b7ec056">

## Reason of using log
- Even in SGD, amount of multiplication is still big. By that even the value is not close to 0, multiply of all will be close to 0.
- Tractable
- For eliminating exponential function

## MLE --> Binary Entropy (Classification)  

<details>
<summary> Difference with LS </summary>

```
link function generating hypothesis value is different 
```
<img width="456" alt="image" src="https://github.com/user-attachments/assets/2c203edc-1eb8-495f-a701-c19048acdc31">
</details>
 



## Newton's Method
- When?
    small low-dimention dataset   

- Learning rate is fixed unlike SGD.

## Optimization Method Summary

|Method|Compute per Step|Number of Steps|
|:---:|:---:|:---:|
|SGD|Low|High|
|Minibatch SGD|Middle|Middle|
|GD|Higher than SGD|Can't say it will be lower than SGD|
|Newton|High|Low|

<details>
<summary>In classical stats, </summary>

```
d is small, n is often small, exact parameters matter
```
</details>

<details>
<summary>In modern ML, </summary>

```
d is huge, n is huge, parameters used only for prediction(Individual parameters don't have such meanings)
```
</details>
