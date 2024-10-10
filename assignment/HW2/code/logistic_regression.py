import numpy as np


class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, batch_size, epochs, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        y : (N, )
        batch_size : (int) # of minibatch size
        epochs : (int) # of training epoch to execute
        lr : (float), learning rate
        optimizer : (Class) optimizer to use

        Returns:
        None

        Description:
        Given training data, hyperparameters and optimizer, execute training procedure.
        Weight should be updated by minibatch (not the whole data at a time)
        The procedure for one epoch is as follows:
        - For each minibatch
            - Compute the probability of each class for data and the loss
            - Compute the gradient of weight
            - Update weight using optimizer

        * loss of one epoch = refer to the loss function in the instruction.
        """

        num_data, num_feat = x.shape
        num_batches = int(np.ceil(num_data / batch_size))

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for b in range(num_batches):
                ed = min(num_data, (b + 1) * batch_size)
                batch_x = x[b * batch_size: ed]
                batch_y = y[b * batch_size: ed]

                prob, loss = self.forward(batch_x, batch_y)
                
                grad = self.compute_grad(batch_x, batch_y, self.W, prob)

                # Update Weights
                self.W = optimizer.update(self.W, grad, lr)

                epoch_loss += loss

    def forward(self, x, y):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data 
        y : (N, ), label for each data

        Returns:
        logits: (N, 1), logit for N data
        loss : float, loss for N input

        Description:
        Given N data and their labels, compute probability distribution and loss.

        Hint:
        For the log function, add epsilon (eps) for numerical stability in log.
        e.g., np.log(logits) -> np.log(logits + eps)
        """

        num_data, num_feat = x.shape

        y = np.expand_dims(y, axis=1)
        
        logits = None
        loss = 0.0
        eps = 1e-10

        h_x = 0
        if logits is None:
            logits = np.array([])

        for i in range(num_data):
            logit = np.sum(np.transpose(self.W) @ x[i])

            logits = np.append(logits, logit)
            h_x = self._sigmoid(logit)
            
            if y[i] == 1:       # cross entropy를 분리시켜서 y가 1인 경우 Positive Sample
                loss -= np.log( h_x +eps)
            elif y[i] == 0:  # y가 0인 경우 Nagative Sample
                loss -= np.log( 1- h_x +eps)

        loss /= num_data

        logits = np.transpose(logits)

        # ============================================================

        return logits, loss
    
    def compute_grad(self, x, y, weight, logit):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data
        y : (N, ), label for each data
        weight : (D, 1), Weight matrix of classifier
        logit: (N, 1), logits for N data

        Returns:
        the gradient of weight: (D, 1), Gradient of weight to be applied (dL/dW)

        Description:
        Given input, label, weight, logit, compute the gradient of weight.
        """

        num_data, num_feat = x.shape

        
        y = np.expand_dims(y, axis=1)
        
        grad_weight = np.zeros_like(weight)

        # ========================= EDIT HERE ========================


        for i in range(num_data):
            h_x = self._sigmoid(logit[i])
            x_value = x[i].reshape(x.shape[1],1)
            grad_weight += (h_x - y[i]) * x_value /num_data

        # ============================================================

        return grad_weight
    
    def _sigmoid(self, x):
        """
        Inputs:
        x : (N, C), score before sigmoid

        Returns:
        sigmoid : (same shape with x), applied sigmoid.

        Description:
        Given an input x, apply the sigmoid function.
        """

        sigmoid = None
        eps = 1e-10
        
        sigmoid = 1/ (1 + np.exp(-x+eps))
        return sigmoid

    def eval(self, x, threshold=0.5):
        """
        Inputs:
        x : (N, D), input data

        Returns:
        pred : (N, ), predicted label for N test data
        prob : (N, 1), predicted logit for N test data

        Description:
        Given N test data, compute logits and make predictions for each data with the given threshold.
        """

        pred = None
        prob = None

        if pred is None:
            pred = np.array([])
        if prob is None:
            prob = np.array([])

        for i in range(x.shape[0]):

            logits = np.sum(np.transpose(self.W) @ x[i])
            #print(logits)
            h_x = self._sigmoid(logits)

            if h_x > threshold:
                pred = np.append(pred, 1)
            elif h_x < threshold:
                pred = np.append(pred, 0)

            prob = np.append(prob , logits)

        return pred, prob
