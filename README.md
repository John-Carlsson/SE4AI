# SE4AI
This is repository for the project within the course Software Engineering for Artificial intelligence. This group consist of 6 Erasmus students at the University of Salerno


Facial detection: 

Some documentation for nnmodel.py: 

This file is ment to act as a type of playground for trying out different models for facial emotion detection. 

We have the class NNmodel that contains some fundamental functions for training NN for facial emotion detection. 
Sofar only one (sequential) model is implemented, which provides a 66% accuracy on the validation set. This is 
quite good, since our expectation is an accuracy of 70%. 

train_model() saves the trained model and returns the path of the trained model.
[Note: training this model takes approx. 30 min.]

Dataset: https://www.kaggle.com/datasets/deadskull7/fer2013

TODO:
  - Documentation missing
  - Write discriptions for the functions
  - cleanup package imports 
  - Implement a efficient file reader, i.e. a way to import different datasets and process these sets.
  - Make predictions on the test-data (visulisation)
  - [maybe split the model and training into two different functions] 
  
  
  
  
  
  

Implemented Model(s): 




Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 46, 46, 256)       2560      
                                                                 
 batch_normalization (BatchN  (None, 46, 46, 256)      1024      
 ormalization)                                                   
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 46, 46, 256)       0         
                                                                 
 conv2d_1 (Conv2D)           (None, 46, 46, 256)       590080    
                                                                 
 batch_normalization_1 (Batc  (None, 46, 46, 256)      1024      
 hNormalization)                                                 
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 46, 46, 256)       0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 23, 23, 256)      0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 23, 23, 256)       590080    
                                                                 
 batch_normalization_2 (Batc  (None, 23, 23, 256)      1024      
 hNormalization)                                                 
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 23, 23, 256)       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 23, 23, 256)       590080    
                                                                 
 batch_normalization_3 (Batc  (None, 23, 23, 256)      1024      
 hNormalization)                                                 
                                                                 
 leaky_re_lu_3 (LeakyReLU)   (None, 23, 23, 256)       0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 11, 11, 256)      0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 11, 11, 128)       295040    
                                                                 
 batch_normalization_4 (Batc  (None, 11, 11, 128)      512       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_4 (LeakyReLU)   (None, 11, 11, 128)       0         
                                                                 
 conv2d_5 (Conv2D)           (None, 11, 11, 128)       147584    
                                                                 
 batch_normalization_5 (Batc  (None, 11, 11, 128)      512       
 hNormalization)                                                 
                                                                 
 leaky_re_lu_5 (LeakyReLU)   (None, 11, 11, 128)       0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 5, 5, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 3200)              0         
                                                                 
 dense (Dense)               (None, 256)               819456    
                                                                 
 batch_normalization_6 (Batc  (None, 256)              1024      
 hNormalization)                                                 
                                                                 
 leaky_re_lu_6 (LeakyReLU)   (None, 256)               0         
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 batch_normalization_7 (Batc  (None, 128)              512       
 hNormalization)                                                 
                                                                 
 activation (Activation)     (None, 128)               0         
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 7)                 903       
                                                                 
=================================================================
Total params: 3,075,335
Trainable params: 3,072,007
Non-trainable params: 3,328
-----------------------------------------------------------------

113/113 [==============================] - 65s 576ms/step - loss: 0.7499 - accuracy: 0.7165 - val_loss: 1.0131 - val_accuracy: 0.6417






