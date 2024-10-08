## Exponential Family  
<img width="543" alt="image" src="https://github.com/user-attachments/assets/c049b5a1-a1dc-46a1-8923-eb2f235c903e">  

- b(y) the base measure depends on y.  
- a(η) the log partition function is given as sum of all probability is 1.  

<details>
<summary>Why do we care about exponential family?</summary>
<img width="573" alt="image" src="https://github.com/user-attachments/assets/98373475-b7ff-496f-9f0d-2628dd3eba36">  


By log partition function, inference of expectation and variance is settled.  

<img width="479" alt="image" src="https://github.com/user-attachments/assets/7b787d88-6cfd-4b60-afdd-a86675ff97e8">  

</details>

## Generalized Linear Models(GLS)

### Pick a link function (and distribution) based on target's type.
<img width="771" alt="image" src="https://github.com/user-attachments/assets/efe6c1bf-8a6b-4423-9a2d-28adb5dc8ea2">

<img width="806" alt="image" src="https://github.com/user-attachments/assets/37a115e4-c019-489e-a173-d9e8c96e41e6">

## Multiclass

- Using one-hot vectors  
- Using softmax function as our link function  
- For general k, a probability estimate for any k-1 class determines the other class  
- Using CrossEntropy as log likelihood function  
<img width="659" alt="image" src="https://github.com/user-attachments/assets/f09f6b25-fef0-4f25-b9c0-0237f7667924">  