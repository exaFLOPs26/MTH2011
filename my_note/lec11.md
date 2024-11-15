## (Tuesday) Unsupervised Learning

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
        <img width="450" alt="image" src="https://github.com/user-attachments/assets/626e0920-05be-4eb5-846a-fd73eebc84d2">
        <img width="450" alt="image" src="https://github.com/user-attachments/assets/91b362eb-c7c3-4daf-8866-95bb73429776">
    - Maximization Step
        - Update the parameters
        <img width="525" alt="image" src="https://github.com/user-attachments/assets/510f6d09-7c16-466f-84c4-246ef8b3a6e6">

### Convex & Jensen's Inequality

<img width="552" alt="image" src="https://github.com/user-attachments/assets/10438c7d-c6dc-42db-9daa-a2a729fb9cf2">
<img width="518" alt="image" src="https://github.com/user-attachments/assets/29563c02-d34c-4deb-b082-0480a6168aa9">
<img width="588" alt="image" src="https://github.com/user-attachments/assets/a0a90120-bc88-491f-a068-755c8ec992b6">

### EM Algorithm as MLE
<img width="588" alt="image" src="https://github.com/user-attachments/assets/6a3515b2-ba03-44c9-bcda-488e070f9142">
<img width="588" alt="image" src="https://github.com/user-attachments/assets/ab4f9b69-5b68-4cf6-8fa1-a274d8954acb">
<img width="526" alt="image" src="https://github.com/user-attachments/assets/e726fb9e-0b9c-4717-8ca0-4dbbadd785ef">


## (Thursday) Evaluatoin Metrics
mnist -data augmentation 