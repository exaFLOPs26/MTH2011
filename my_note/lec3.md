수학적인 사고는 연역적으로 assumption을 바탕으로 풀어나가는 것 이를 통해 세상을 조금 더 설명력있게 but 현실과는 조금 떨어질 수도...

## OLS in MLE view

### Steps
1. Assume our hypothesis as linear model w\ noise  
    <img width="147" alt="image" src="https://github.com/user-attachments/assets/0f3cea65-f5bb-4a5f-8b4a-f1f76fe64218">  

2. Assumptions on noise  
    - noise is unbiased  
    - errors for different points are iid(independent and identically distributed)  
    <img width="431" alt="image" src="https://github.com/user-attachments/assets/b1375f35-c4cc-4f6e-924f-ad92da1b2ff3">
3. Use Gaussian distribution for our model and parameter  
    <img width="431" alt="image" src="https://github.com/user-attachments/assets/1a69f09a-38d2-4714-84d0-9467b330545e">

4. By defining (log) likelihood, solving least squares is solving a maximum likelihood problem for a particular probabilistic model. 
    <img width="483" alt="image" src="https://github.com/user-attachments/assets/180fe6f9-4558-466b-b48b-e82b2b7ec056">
