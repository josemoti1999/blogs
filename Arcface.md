# Understanding Basic Concepts for ArcFace Loss

The main goal of this work is to give a beginner summary of Arcface loss function. This notebook was written with my limited knowledge on the topic. Suggestions are highly appreciated.
 
## Softmax Loss

So I will start with the softmax loss function. This term can indeed cause some confusion because softmax is an activation function and not a loss function. Softmax loss is just categorical cross entropy loss done after softmax activation and is given by  

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{z_{true}}}{\sum_{j=1}^{n} e^{z_j}}$$

where N is the batch size, n is the number of classes present, $e^{z_{true}}$ is the softmax probability of the true class and it is divided by the sum of the softmax probabilities of the remaining classes.

The main problem with softmax loss is that it does not make use of the embeddings (the output before the final layer) in such a way that there is more similarity between intra class labels and less between inter class labels. Even though softmax loss can perform well in closed-set classification task like the imagenet task where all the classes are known it fails to give similar performance with open-set classification tasks which have several unknown classes like Face Detection Tasks or Landmark Recognition tasks to name a few. This is where SphereFace, CosFace or ArcFace losses come out to be useful. We will just focus on ArcFace loss now. **Arcface optimizes the geodesic distance margin between the classes.** Geodesic distance is the shortest distance between two points while going along with a surface. We will come back to this later.

Now let us try to understand more on the softmax loss function. In the loss function given above, $z_j$ can be described with the weights, biases and embeddings of the previous layer as 

$$z_j =  W_j^Tx_i + b_j$$ 

where $x_i$ is the embedding vector of a single image and $W_j$ and $b_j$ are the corresponding class columns. Let us take the embedding vector as $512$ dimensional and 1000 classes. So W is $512*1000$ dimensional and bias is 1000-D. Checking the matrix multiplications $(1000*512)*(512*1)$ --> $(1000*1)$.  So $W_j$ is actually the $512*1$ column which is responsible for the j-th logit. 

With this, we can rewrite the softmax function as 

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{W_{true}^Tx_i + b_{true}}}{\sum_{j=1}^{n} e^{W_j^Tx_i + b_j}}$$

Now comes the interesting part.. So it is clear that both $W_j$ and $x_i$ are 512-dimensional(512\*1). The dimension of $W_j^T$ is (1, 512) and $x_i$ is (512, 1). So taking their cross product is actually the dot product of $W_j$ (512\*1) dimensional and $x_i$ (512\*1) dimensional. ie here

$$W_j^T \times x_i = W_j \cdot x_i = \lVert{W_j}\rVert \lVert{x_i}\rVert cos\theta_j$$ 

where $cos \theta_j$ is the angle between $x_i$ and $W_j$. 

Ignoring bias term the softmax loss can now be rewritten as 

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{\lVert{W_{true}}\rVert \lVert{x_i}\rVert cos\theta_{true}}}{\sum_{j=1}^{n} e^{\lVert{W_j}\rVert \lVert{x_i}\rVert cos\theta_j}}$$

Now $W_j$ is l2-normalized making $\lVert W_j \rVert = 1$, for all $j \in (1,n)$. Similarly $x_i$ is l2-normalized and multiplied by a constant s making $\lVert x_i \rVert = s$. So now again the loss can be rewritten as

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{s \cdot cos\theta_{true}}}{\sum_{j=1}^{n} e^{s \cdot cos\theta_j}}$$

So now $x_i$ is a vector that lies on the surface of 512-dimensional hypersphere with a radius of s. And $W_j$ lies on the surface of similar dimensional hypersphere but with radius 1. Inorder to minimize the loss function we need maximize $cos\, \theta_{true}$ where $cos \theta_{true}$ is the angle between weight of the true logit and the embedding and minimize the remaining $cos\, \theta_j$ for all $j \in (1,n) and j \ne y_i$ where $y_i$ is the true logit. So after training the logit weight vectors would be fixed and softmax does the classification based on the distance of the embedding of the image to be predicted from the each of these logit weight vectors. The more close $x_i$ comes near $W_k$ the more chance the k-th class is predicted. This scenario of 2-dimensional embeddings with 8 classes are shown below . See (a).

![image.png](https://miro.medium.com/max/774/1*j7ikHwBZvLzShQWq8Nf6tw.png)

Now we get a clear picture of why we use use geodesic distance because the embedding vecturs lie only in hypersphere of lets say for example 512 dimensional and while measuring the distance we should measure the distance along this hypersphere surface. As can be seen from the diagram above the problem is actually the lack of high interclass geodesic distance compared to that of intraclass distance. So while predicting on open datasets with softmax loss there is a high chance that it gets a good confidence in any 1 of the given classes. This is what Arcface solves. As seen in (b) arcface brings up a larger geodesic distance between the centre of each classes compared to its intra class distance. And that too in few simple steps. We will see that now.

## ArcFace Loss
The softmax function can be slightly modified as 

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{s \cdot cos\theta_{y_i}}}{e^{s \cdot cos\theta_{y_i}} + {\sum_{j=1, j \ne y_i}^{n} e^{s \cdot cos\theta_j}}}$$

where $y_i$ is the true logit. Now we insert an angular margin between $W_{y_i}$ and $x_i$. This makes the new angle between true logit and embedding to be $(\theta_{y_i} + m)$. This helps in penalizing the embedding vectors that goes far and help in bringing the embedding features of a certain class come more closer. This leads to smaller intraclass geodesic distance compared to interclass geodesic destance as shown in (b). So our final loss function is

$$L_1 = -\frac{1}{N} \sum_{i=1}^{N} \frac{e^{s \cdot cos{(\theta_{y_i} + m)}}}{e^{s \cdot cos{(\theta_{y_i} + m)}} + {\sum_{j=1, j \ne y_i}^{n} e^{s \cdot cos\theta_j}}}$$

Note that the angular margin penalty is given only to the true logit. Also for a given image the logit confidence scores may not add up to 0. This can be understood in a similar way to what regularization does to overfitting. Adding an the l2 distance/ Euclidean distance of the weight sto the loss function makes the model select small weights which helps to decrease the variance. Similarly having an angular penalty makes the model to help make the embedding vectors from a class much closer to each other because more the angle, more the cos penalty. 


Paper: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://support.west-wind.com)
