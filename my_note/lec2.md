## Linear Regression

- Steps  
    1. Hypothesis is a function of X and y  
    2. Restrict the function into linear function(affine function) w\ parameters & features  
    3. By using OLS choose the best parameter, making it close to the data

- Adding more features can make linear model bad.
- At least d(# of feature)+1 data should be collected.

## Gradient Descent  

- Goal: optimize loss function

- Update rule
<img width="282" alt="image" src="https://github.com/user-attachments/assets/35bd7cb0-269a-4ba4-b93d-770a70b84c6a">

- Using all n data points is not alligned with the concept of "Big Data".]

## Stochastic GD

- minibatch: |B| = 1
- smaller learning rate ~ larger batch size
- Smaller B implies a lower quality approximation of the gradient (higher variance)  
- Nevertheless, it may actually converge faster! (Case where the dataset has many copies of the same point–extreme, but lots of redundancy) I In practice, choose B proportional to what works well on modern parallel hardware (GPUs).


## Normal equations

<img width="205" alt="image" src="https://github.com/user-attachments/assets/7af73f3a-5263-47d1-a413-fadbd650d342">

- Normal equation is usally used at small amount of data situation (eteration 반복 필요 없을때-그래서 과거에서는 어떤 feature를 사용할지가 매우 중요)

- Design matrix  
<img width="421" alt="image" src="https://github.com/user-attachments/assets/36e369b6-f5fd-497b-af2c-b8c75e8fb3fe">

- Assuming (X^X)^(-1) exist, but it might not?! (특히 feature 많은 상황-feature 간 dependence 클 때)  
- (X^X) becomes a big matrix and finding it's inverse is very expensive.  
- Convex is guaranteed so when normal equation becomes 0 is the minimum. Because loss function's second derivative (derivative of normal equation)is positive semi definite.  