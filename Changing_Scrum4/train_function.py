from tensorflow import keras
import time

def first_stage(model,train_dataset,val_dataset,num_samples,type='multi_label',max_epoch=5,load_weight = False,savepath = './save/f1_checkpoint',loadpath = './save/f1_checkpoint'):
    """Stage1 and Stage3 Training Loop
        In this loop, it just like normal model training loop with constant learning
        output is loss of all samples in each epoch for seeing loss behavior
    """

    ################## Initial optimizer, loss, metric ########################
    optimizer = keras.optimizers.SGD(
                    learning_rate=0.01,
                    momentum=0.9,
                    decay = 5e-4
                )
    if type == 'multi_label':
        loss_fn = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        model.compile(optimizer=optimizer, loss=loss_fn,metrics=['binary_accuracy'])
        train_acc_metric = keras.metrics.BinaryAccuracy()
        val_acc_metric = keras.metrics.BinaryAccuracy()
        savepath = './save/s2_checkpoint_nih'
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        train_acc_metric = keras.metrics.CategoricalAccuracy()
        val_acc_metric = keras.metrics.CategoricalAccuracy()
        model.compile(optimizer=optimizer, loss=loss_fn,metrics=['categorical_accuracy'])
        savepath = './save/s2_checkpoint_cifar'
    ################## Initial optimizer, loss, metric ########################

    ################## Initial loss tracking variable ########################
    if load_weight == True:
        model.load_weights(savepath)
    example_loss = np.zeros(num_samples)
    all_sample_loss = []
    ################## Initial loss tracking variable ########################

    ################## Create tf.function for training model ########################
    @tf.function
    def train_step(x_batch_train,y_batch_train):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        return loss_value

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
    ################## Create tf.function for training model ########################

    for epoch in range(max_epoch):

        ################## Initial Step ########################
        start_time = time.time()
        indexes = 0

        print("\nStart Epoch : " ,epoch)
        ################## Initial Step ########################

        ################## Training Step ########################
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            loss_value = train_step(x_batch_train,y_batch_train)

            for l in loss_value:
                example_loss[indexes] = float(l)
                # print(example_loss[indexes])
                indexes +=1

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()
        print("Training Time taken: %.2fs" % (time.time() - start_time))
        start_time = time.time()
        ################## Training Step ########################

        ############ Validation Step ####################
        for step,(x_batch_val, y_batch_val) in enumerate(val_dataset):
            test_step(x_batch_val, y_batch_val)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation Time taken: %.2fs" % (time.time() - start_time))
        ############ Validation Step ####################

        ################## Loss Tracking Step #########################
        ## save loss of each samples
        all_sample_loss.append(example_loss.tolist().copy())
        save_json(example_loss.tolist().copy(),'./save_loss/loss_f1_'+str(epoch))

        model.save_weights(savepath)

    return all_sample_loss

def second_stage(model,train_dataset,val_dataset,num_samples,noise_or_not=None,type='multi_label',max_epoch=5,save_weight = False,load_weight = True,r1=0.01,r2=0.001,cy_loop = 10,forget_rate=0.1):
    """Stage2 Training Loop (main O2U_net)
        In this loop, it will train model with cyclical learning rate in linear form where r1 is maximum lr, r2 is minimum lr, cy_loop is number of epoch for calculating r1 to r2
        forget_rate is rank of cut off noise use in creating filter_mask
        noise_or_not is ground truth which is boolean of each samples which True mean it is true label and False mean it is noisy label
    """
    
    ################## Initial optimizer, loss, metric ########################
    optimizer = keras.optimizers.SGD(
                    learning_rate=0.01,
                    momentum=0.9,
                    decay = 5e-4
                )
    if type == 'multi_label':
        loss_fn = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        model.compile(optimizer=optimizer, loss=loss_fn,metrics=['binary_accuracy'])
        train_acc_metric = keras.metrics.BinaryAccuracy()
        val_acc_metric = keras.metrics.BinaryAccuracy()
        savepath = './save/s2_checkpoint_nih'
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        train_acc_metric = keras.metrics.CategoricalAccuracy()
        val_acc_metric = keras.metrics.CategoricalAccuracy()
        model.compile(optimizer=optimizer, loss=loss_fn,metrics=['categorical_accuracy'])
        savepath = './save/s2_checkpoint_cifar'
    ################## Initial optimizer, loss, metric ########################

    ################## Initial loss tracking variable ########################
    if load_weight == True:
        model.load_weights(savepath)
    moving_loss_dic = np.zeros(num_samples)
    globals_loss = 0
    example_loss = np.zeros(num_samples)
    all_sample_loss = []
    ################## Initial loss tracking variable ########################

    ################## Create tf.function for training model ########################
    @tf.function
    def train_step(x_batch_train,y_batch_train):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        return loss_value

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
    ################## Create tf.function for training model ########################

    for epoch in range(max_epoch):

        ################## Initial Step ########################
        start_time = time.time()
        indexes = 0
        ## calculate lr in linear equation from 0.0091 to 0.001 and delta == 0.0009
        t = (epoch % cy_loop) / float((cy_loop)-1)
        lr = (1 - t) * r1 + t * r2
        optimizer.learning_rate = lr
        print("\nStart Epoch : " ,epoch,"learning rate : ",lr)
        ################## Initial Step ########################

        ################## Training Step ########################
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            loss_value = train_step(x_batch_train,y_batch_train)

            for l in loss_value:
                example_loss[indexes] = float(l)
                # print(example_loss[indexes])
                indexes +=1

            globals_loss += loss_value.numpy().sum()
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()
        print("Training Time taken: %.2fs" % (time.time() - start_time))
        start_time = time.time()
        ################## Training Step ########################

        ############ Validation Step ####################
        for step,(x_batch_val, y_batch_val) in enumerate(val_dataset):
            test_step(x_batch_val, y_batch_val)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation Time taken: %.2fs" % (time.time() - start_time))
        ############ Validation Step ####################

        ################## Loss Tracking Step #########################
        ## save loss of each samples
        all_sample_loss.append(example_loss.tolist().copy())
        save_json(example_loss.tolist().copy(),'./save_loss/loss_s2_'+str(epoch))

        ## normalize all loss with average loss
        example_loss = example_loss - example_loss.mean()

        ## stack loss of each epoch
        moving_loss_dic=moving_loss_dic+example_loss

        ## return indices of sorted array
        ind_1_sorted = np.argsort(moving_loss_dic)
        ## sort total by use sorted indices array
        loss_1_sorted = moving_loss_dic[ind_1_sorted]

        ## set nth rank to keep
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ## create filter_mask
        mask = np.ones_like(example_loss,dtype=np.float32)
        if noise_or_not is not None:
            ## find accuracy of noise detection
            noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)

            ## set top n% rank to zero (for cleansing noise)
            mask[ind_1_sorted[num_remember:]]=0

            ## find top 0.1 noise accuracy
            top_accuracy_rm=int(0.9 * len(loss_1_sorted))
            top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)
            print ( "noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)
        ################## Loss Tracking Step #########################

    if save_weight == True:
            model.save_weights(savepath)

    return all_sample_loss,mask,ind_1_sorted
