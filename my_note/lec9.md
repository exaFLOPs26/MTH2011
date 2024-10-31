## (10/29) Deep Laerning

Deep Learning is still in a form of GLM.

Goal of AI is efficiency. And unconsciousness might help.
Mathmetical expression of unconsciousness. No need to make complex optimization

- Why use ReLU than sigmoid, tanh?
    - With sigmoid and tanh, we can't discriminate learning for some point.
    - Similar to ReLU as negative signal results the same, we can use leaky ReLU.

Feature mapping is good for non-linear but how?!
We don't know. So just give it to Neural Net and optimize it.

In math, want to find the function. But Neural Net is doing good. Because of stacking(composition funciton). And by stacking NN can find the derived features(variables) by input data.

Most weekness of NN is interpretaion.

### Residual connection
 
<img width="751" alt="image" src="https://github.com/user-attachments/assets/e2979f16-f363-4019-ad51-51aa7cab2e4f">


## (10/31) Backpropagation

> Backpropagation is just chain rule of composition functions

<img width="538" alt="image" src="https://github.com/user-attachments/assets/6eb71262-c01c-4b84-bd4a-baa300d770c2">
