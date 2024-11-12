## Unsupervised Learning

- We implicitly think Euclidean distance as similarity
- Simple is best
- y, which is label, is gone and z, latent variable, comes in.

### k-Means

- Bad
    - Decision boundary is not specific
        - Soft assign (GMM)
    - Unbalance between clusters
    - Sensitive on initialization
        - Enough distance between initialized points

### Gaussian Mixture Model (GMM)

- Compare with k-means, a soft (probabilistic) assignment
- If you want to do it softly ~ stochastic(probabilistic) view
- GMM parameter by EM Algorithm
    - Expectation Step
        - "Guess the latent values of z^(i)" for each point i = 1,...,n
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/626e0920-05be-4eb5-846a-fd73eebc84d2">
        <img width="300" alt="image" src="https://github.com/user-attachments/assets/91b362eb-c7c3-4daf-8866-95bb73429776">
    - Maximization Step
        - Update the parameters
