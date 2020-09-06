'''
Gyanendra Sharma
05/10/2017
Deep Learning Final Project
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
#load files
npzfile = np.load("train_and_val.npz")
train_eye_left = npzfile["train_eye_left"]
train_eye_right = npzfile["train_eye_right"]
train_face = npzfile["train_face"]
train_face_mask = npzfile["train_face_mask"]
train_y = npzfile["train_y"]

val_eye_left = npzfile["val_eye_left"]
val_eye_right = npzfile["val_eye_right"]
val_face = npzfile["val_face"]
val_face_mask = npzfile["val_face_mask"]
val_y = npzfile["val_y"]

#converting to float, scaling and normalizing
train_eye_left,val_eye_left = train_eye_left.astype('f'),val_eye_left.astype('f')
train_eye_left,val_eye_left = train_eye_left/255.0,val_eye_left/255.0
train_eye_left,val_eye_left = train_eye_left - train_eye_left.mean(),val_eye_left - val_eye_left.mean()

train_eye_right,val_eye_right = train_eye_right.astype('f'),val_eye_right.astype('f')
train_eye_right,val_eye_right = train_eye_right/255.0,val_eye_right/255.0
train_eye_right,val_eye_right = train_eye_right - train_eye_right.mean(),val_eye_right - val_eye_right.mean()

train_face,val_face = train_face.astype('f'),val_face.astype('f')
train_face,val_face = train_face/255.0,val_face/255.0
train_face,val_face = train_face - train_face.mean(),val_face - val_face.mean()

train_face_mask,val_face_mask = train_face_mask.astype('f'),val_face_mask.astype('f')
train_face_mask,val_face_mask = train_face_mask/255.0,val_face_mask/255.0
train_face_mask,val_face_mask = train_face_mask - train_face_mask.mean(),val_face_mask - val_face_mask.mean()

print (train_eye_left.shape)
print (train_face.shape)
print (train_face_mask.shape)
print (train_y.shape)

#function to get batch size samples
def getSamples(batch_size,flag):
    if (flag==0):
        number = 48000
        xa = train_eye_left
        xb = train_eye_right
        xc = train_face
        xd = train_face_mask
        y = train_y
    else:
        number = 5000
        xa = val_eye_left
        xb = val_eye_right
        xc = val_face
        xd = val_face_mask
        y = val_y
    index_list = np.random.randint(1,number,batch_size)
    x1 = np.zeros([batch_size,64,64,3])
    x2 = np.zeros([batch_size,64,64,3])
    x3 = np.zeros([batch_size,64,64,3])
    x4 = np.zeros([batch_size,25,25])
    y1 = np.zeros([batch_size,2])
    count = 0
    for i in index_list:
        x1[count,:,:,:] = xa[i,:,:,:]
        x2[count,:,:,:] = xb[i,:,:,:]
        x3[count,:,:,:] = xc[i,:,:,:]
        x4[count,:,:] = xd[i,:,:]
        y1[count,:]=y[i,:]
        count=count+1
    return x1,x2,x3,x4,y1

#defining parameters
lr_rate = 0.001
tr_iter = 5
batch_size = 512
print_in = 1

im_dim1 = 64
im_dim2 = 25
channels = 3
classes = 2
#path for the events to be saved for tensorboard visualization
logs_path = 'C:/Users/Admin/Desktop/events'
#defining placeholders for each of the input and output
eye_left = tf.placeholder(tf.float32, [None,im_dim1,im_dim1,3],name='left_eye')#left eye
eye_right = tf.placeholder(tf.float32, [None,im_dim1,im_dim1,3],name='right_eye')#right eye
face = tf.placeholder(tf.float32, [None,im_dim1,im_dim1,3],name='face')#face
face_mask = tf.placeholder(tf.float32, [None,im_dim2,im_dim2],name='face_mask')#mask
label_y = tf.placeholder(tf.float32, [None, classes],name='y_label')
#defining shapes convolution layer 1, 2 and 3 to be used for left eye, right eye and face
#although shape is the same, the weights are shared only between left and right eye
#whereas face has independent weights
w1_shape_e = [5,5,3,32]
w2_shape_e = [5, 5, 32, 32]
w3_shape_e = [3, 3, 32, 64]

#setting up placeholders for eye 1 and eye 2 with shared weights
# 5x5 conv, 3 input, 32 predict_opputs
W1_e  = tf.get_variable('eyes_w1', shape=w1_shape_e, initializer=tf.contrib.layers.xavier_initializer())
# 5x5 conv, 32 inputs, 32 predict_opputs
W2_e = tf.get_variable('eyes_w2', shape=w2_shape_e, initializer=tf.contrib.layers.xavier_initializer())
# 3x3 conv, 32 inputs, 64 predict_opputs
W3_e = tf.get_variable('eyes_w3', shape=w3_shape_e, initializer=tf.contrib.layers.xavier_initializer())
#bias for each of the conv layers
b1_e = tf.Variable(tf.random_normal([32]),name='eyes_b1')
b2_e = tf.Variable(tf.random_normal([32]),name='eyes_b2')
b3_e = tf.Variable(tf.random_normal([64]),name='eyes_b3')

#setting up placeholders for face
#we use the same dimension for weights matrices for face input too
# 5x5 conv, 3 input, 32 predict_opputs
W1_f  = tf.get_variable('face_w1', shape=w1_shape_e, initializer=tf.contrib.layers.xavier_initializer())
# 5x5 conv, 32 inputs, 32 predict_opputs
W2_f = tf.get_variable('face_w2', shape=w2_shape_e, initializer=tf.contrib.layers.xavier_initializer())
# 3x3 conv, 32 inputs, 64 predict_opputs
W3_f = tf.get_variable('face_w3', shape=w3_shape_e, initializer=tf.contrib.layers.xavier_initializer())
#bias for each of the conv layers for face
b1_f = tf.Variable(tf.random_normal([32]),name='face_b1')
b2_f = tf.Variable(tf.random_normal([32]),name='face_b2')
b3_f = tf.Variable(tf.random_normal([64]),name='face_b3')

#first fully connected layer for eye1 and eye2
e1_long = [11*11*64,256]
#combined fully connected layer for eye1 and eye2
ec_long = [512,256]
#first fully connected layer for the face
f_long = [11*11*64,256]
#second fully connected layer for face
f_long2 = [256,256]
#mask fully connected layer 1 shape
m_long = [25*25,256]
#mask fully connected layer 2 shape
m_long2 = [256,128]

#dense layers for individual eyes
#layer 1
dense_eye = tf.get_variable('dense_eye', shape=e1_long, initializer=tf.contrib.layers.xavier_initializer())
dense_eye2 = tf.Variable(tf.random_normal([256]),name='face_dense_b1')
#dense input layers for face
#layer 1
dense_face = tf.get_variable('dense_face', shape=f_long, initializer=tf.contrib.layers.xavier_initializer())
dense_face2 = tf.Variable(tf.random_normal([256]),name='face_dense_b1')
#layer 2
dense_face_l2 = tf.get_variable('dense_face_l2', shape=f_long2, initializer=tf.contrib.layers.xavier_initializer())
dense_face_l2_2 = tf.Variable(tf.random_normal([256]),name='face_dense_b2')

#setting up placeholders for mask fully connected layer
#layer 1
dense_mask = tf.get_variable('mask_dense_l1', shape=m_long, initializer=tf.contrib.layers.xavier_initializer())
dense_mask2 = tf.Variable(tf.random_normal([256]),name='mask_dense_b1')
#layer2
dense_mask_l2 = tf.get_variable('mask_dense_l2', shape=m_long2, initializer=tf.contrib.layers.xavier_initializer())
dense_mask_l2_2 = tf.Variable(tf.random_normal([128]),name='mask_dense_l2_2')

#setting up variables for combined eyes
eye_fc_shape = [512,256]
eye_fc = tf.get_variable('eye_fc_w1', shape=eye_fc_shape, initializer=tf.contrib.layers.xavier_initializer())
eye_fc2 = tf.Variable(tf.random_normal([256]),name='eye_dense_b1')
#setting up variables and placeholders for the combined fully connected layers
#left eye right eye and face have 3*3*64 while mask and face have 128, setting up output fc layer to 512
full_conn_shape = [2*256 + 128, 512]
dense_final = tf.get_variable('dense_final_w1', shape=full_conn_shape, initializer=tf.contrib.layers.xavier_initializer())
dense_final2 = tf.Variable(tf.random_normal([512]),name='dense_final_b1')

#from hidden layer of 512 nodes to 256 nodes
output_layer_shape = [512,256]
output_final = tf.get_variable('out_layer_w1', shape=output_layer_shape, initializer=tf.contrib.layers.xavier_initializer())
output_final2 = tf.Variable(tf.random_normal([256]),name='out_layer_b1')

#from 256 nodes to output class of size 2
output_layer2_shape = [256,classes]
output_final_l2 = tf.get_variable('out_layer2_w1', shape=output_layer2_shape, initializer=tf.contrib.layers.xavier_initializer())
output_final_l2_2 = tf.Variable(tf.random_normal([classes]),name='out_layer2_b2')

#function to carry out three layers of convolution for each of the face, right eye and left eye
def convolveCompute(x,w1,w2,w3,b1,b2,b3):
    #reshaping as needed
    x1 = tf.reshape(x,shape=[-1,im_dim1,im_dim1,channels])
    #convolution layer 1
    with tf.name_scope('Conv1'):
        x2 = tf.nn.conv2d(x1,w1,strides=[1,1,1,1],padding='VALID')
        x3 = tf.nn.bias_add(x2,b1)
        conv1 = tf.nn.relu(x3)
        conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #convolution layer 2
    with tf.name_scope('Conv2'):
        m = tf.nn.conv2d(conv1,w2,strides=[1,1,1,1],padding='VALID')
        m = tf.nn.bias_add(m,b2)
        conv2 = tf.nn.relu(m)
        conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #convolution layer 2
    with tf.name_scope('Conv3'):
        p = tf.nn.conv2d(conv2,w3,strides=[1,1,1,1],padding='VALID')
        p = tf.nn.bias_add(p,b3)
        conv3 = tf.nn.relu(p)
    #create flattened layer after the last convolution
    d1 = tf.reshape(conv3, [-1, 11*11*64])#last size is basically 3*3 with 128 filters for both eye inputs and face inputs
    return d1

#Creating the Training model architecture
def training_Model(eye_L, eye_R, face, mask):
    #pass through convolution for left and right eye
    with tf.name_scope('left_eye_convolve'):
        left_eye_conv = convolveCompute(eye_L,W1_e,W2_e,W3_e,b1_e,b2_e,b3_e,)
    with tf.name_scope('left_eye_FC'):
        left_eye_fc = tf.add(tf.matmul(left_eye_conv,dense_eye),dense_eye2)
        left_eye_output = tf.nn.relu(left_eye_fc)
    with tf.name_scope('right_eye_convolve'):
        right_eye_conv = convolveCompute(eye_R,W1_e,W2_e,W3_e,b1_e,b2_e,b3_e)
    with tf.name_scope('right_eye_FC'):
        right_eye_fc = tf.add(tf.matmul(right_eye_conv,dense_eye),dense_eye2)
        right_eye_output = tf.nn.relu(right_eye_fc)
    with tf.name_scope('eye_concat'):
        eyes = tf.concat([left_eye_output,right_eye_output],1)
    with tf.name_scope('eyes_fc'):
        eyes_fc = tf.add(tf.matmul(eyes,eye_fc),eye_fc2)
        eyes_fc_out = tf.nn.relu(eyes_fc)
    
    #pass through convolution for face
    with tf.name_scope('face_convolve'):
        face_conv = convolveCompute(face,W1_f,W2_f,W3_f,b1_f,b2_f,b3_f)
    #pass the face output through a fully connected layer
    with tf.name_scope('face_dense1'):
        face_output = tf.add(tf.matmul(face_conv,dense_face),dense_face2)
        face_output1 = tf.nn.relu(face_output)
    #pass this face output through another fully connected layer
    with tf.name_scope('face_dense2'):
        face_output2 = tf.add(tf.matmul(face_output1,dense_face_l2),dense_face_l2_2)
        face_output3 = tf.nn.relu(face_output2)
    
    #create fully connected layer for face mask
    d2 = tf.reshape(mask, [-1, dense_mask.get_shape().as_list()[0]])
    with tf.name_scope('face_mask_layer'):
        mask_output = tf.add(tf.matmul(d2,dense_mask),dense_mask2)
        mask_output1 = tf.nn.relu(mask_output)
    #fully connected layer 2 for mask
    with tf.name_scope('face_mask_layer2'):
        mask_output2 = tf.add(tf.matmul(mask_output1,dense_mask_l2),dense_mask_l2_2)
        mask_output3 = tf.nn.relu(mask_output2)
    #concatanate all inputs to one input for fully connected computations
    with tf.name_scope('concat'):
        final_full_connect_layer = tf.concat([eyes_fc_out,face_output3,mask_output3],1)
    
    #full connect layer1 to hidden layer
    with tf.name_scope('full_connected1'):
        out1 = tf.add(tf.matmul(final_full_connect_layer,dense_final),dense_final2)
        out2 = tf.nn.relu(out1)
    #hidden layer 1 to hidden layer 2
    with tf.name_scope('full_connected2'):
        out3 = tf.add(tf.matmul(out2,output_final),output_final2)
        out4 = tf.nn.relu(out3)
    #hidden layer 2 to output layer
    with tf.name_scope('outputlayer'):
        final_output = tf.add(tf.matmul(out4,output_final_l2),output_final_l2_2)
    return final_output

	
#construction of the model, predict_op is the predictions
with tf.name_scope('TrainingModel'):
    predict_op = training_Model(eye_left,eye_right,face,face_mask)
    #cost function
    with tf.name_scope('Loss'):
        loss = tf.reduce_sum(tf.pow(predict_op - label_y, 2)) / (2 * batch_size)


#saving the model
#create the collection
tf.get_collection("validation_nodes")
#add stuff to the collection
tf.add_to_collection("validation_nodes", eye_left)
tf.add_to_collection("validation_nodes", eye_right)
tf.add_to_collection("validation_nodes", face)
tf.add_to_collection("validation_nodes", face_mask)
tf.add_to_collection("validation_nodes", predict_op)
#saving the model
saver = tf.train.Saver()
save_path = 'C:/Users/Admin/Desktop/finalproject/out'
save_path2 = 'C:/Users/Admin/Desktop/finalproject/out2'


#optimizing
with tf.name_scope('AdamOptimizer'):
    optimize = tf.train.AdamOptimizer(learning_rate = lr_rate).minimize(loss)
#Model evaluation
#correctness = tf.reduce_mean(tf.squared_difference(predict_op, label_y))
with tf.name_scope('ModelEval'):
    correctness = tf.reduce_mean(tf.pow(tf.reduce_sum(tf.pow(predict_op - label_y, 2),1),0.5))
# Initializing variables
init = tf.global_variables_initializer()
# Create a summary to monitor Loss
tf.summary.scalar("Loss", loss)
# Create a summary to monitor accuracy
tf.summary.scalar("ModelEval", correctness)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
training_err_vals = []
testing_err_vals = []
loss_values = []
min_tst_acc = 100
with tf.Session() as sess:
    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    step = 1
    while step < tr_iter:
        batch_x1,batch_x2,batch_x3,batch_x4,batch_y = getSamples(batch_size,0) 
        _,summary = sess.run([optimize, merged_summary_op], feed_dict={eye_left: batch_x1,eye_right: batch_x2,face: batch_x3,face_mask: batch_x4, label_y: batch_y})
        if step % print_in == 0:
            # Write logs at every iteration
            summary_writer.add_summary(summary, step)
            loss_val, acc = sess.run([loss, correctness], feed_dict={eye_left: batch_x1,eye_right: batch_x2,face: batch_x3,face_mask: batch_x4, label_y: batch_y})
            loss_values.append(loss_val)
            training_err_vals.append(acc)
            t_batchX1,t_batchX2,t_batchX3,t_batchX4,t_batchY = getSamples(batch_size,1)
            test_acc = sess.run(correctness, feed_dict={eye_left: t_batchX1,eye_right: t_batchX2,face: t_batchX3,face_mask: t_batchX4, label_y: t_batchY})
            testing_err_vals.append(test_acc)
            print ("iter no, Loss, Train_acc, Test_acc: ", step, loss_val, acc, test_acc)
            #saving early stopping models to save for better accuracy
            if (acc<2.6 and test_acc<2.6 and test_acc<min_tst_acc):
                saver.save(sess,os.path.join(save_path2, 'my-model'))
                min_tst_acc = test_acc
                lr_rate = lr_rate/10
        step += 1
    print ("Optimization Finished!")
    save_path = saver.save(sess,os.path.join(save_path, 'my-model'))
    #compute final testing accuracy
    #get a batch of 1000..feeding entire dataset gives errors
    t_batchX1,t_batchX2,t_batchX3,t_batchX4,t_batchY = getSamples(1000,1)
    average = sess.run(correctness, feed_dict={eye_left: t_batchX1,eye_right: t_batchX2,face: t_batchX3,face_mask: t_batchX4, label_y: t_batchY})
    print ("Average error for entire Validation set: ",average)

filehandler = open('C:/Users/Admin/Desktop/finalproject/out/tr_tst_loss.txt',"wb")
values_save = [training_err_vals,testing_err_vals,loss_values,average]#to create graph later just in case
pickle.dump(values_save, filehandler, protocol=2)
filehandler.close()
fig1 = plt.gcf()
plt.plot(training_err_vals,'r--')
plt.plot(testing_err_vals,'g--')
plt.ylabel('training/testing accuracy')
plt.draw()
fig1.savefig('C:/Users/Admin/Desktop/finalproject/out/tr_and_tst.png')
plt.close()
plt.plot(loss_values,'b--')
plt.ylabel('cost')
fig2 = plt.gcf()
plt.draw()
fig2.savefig('C:/Users/Admin/Desktop/finalproject/out/loss.png')
plt.close()



































