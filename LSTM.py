import numpy as np
import tensorflow as tf
import data_processing
import pandas as pd
from sklearn.metrics import mean_absolute_error
import seismic_prediction as context

train_csv_path = context.test_csv_path
sample_submission_path = context.sample_submission_path
test_csv_path = context.test_csv_path
csv_piece_path = context.traning_csv_piece_path
saving_path = context.saving_path
column1_name = context.csv_data_name
column2_name = context.csv_time_name
on_cloud = context.on_cloud
csv_piece_path = context.traning_csv_piece_path

HIDDEN_SIZE = 48                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 150000                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 2000                   # 训练数据个数。
TESTING_EXAMPLES = 500                    # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。

def lstm_model(X, y, is_training):
    # 使用多层的LSTM结构。
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
        for _ in range(NUM_LAYERS)])    

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    model = outputs[:, -1, :]

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    model = tf.contrib.layers.fully_connected(model, 128, activation_fn=None)
    model = tf.contrib.layers.fully_connected(model, 64*64*3, activation_fn=None)
    model = tf.reshape(model, [-1, 64, 64, 3])

    #64*64*3
    model = tf.contrib.layers.conv2d(model, 64, [5, 5], padding='VALID')
    model = tf.contrib.layers.max_pool2d(model, [2, 2], stride=2, padding='VALID')

    #30*30*64
    model = tf.contrib.layers.conv2d(model, 128, [5, 5], padding='VALID')
    model = tf.contrib.layers.max_pool2d(model, [2, 2], stride=2, padding='VALID')

    #13*13*128
    model = tf.contrib.layers.conv2d(model, 256, [4, 4], padding='VALID')
    model = tf.contrib.layers.max_pool2d(model, [2, 2], stride=2, padding='VALID')

    #5*5*256
    model = tf.contrib.layers.conv2d(model, 512, [4, 4], padding='VALID')
    model = tf.contrib.layers.max_pool2d(model, [2, 2], stride=2, padding='VALID')

    #1*1*512
    model = tf.squeeze(model,[1, 2])
    predictions = tf.contrib.layers.fully_connected(model, 1, activation_fn=None)
    predictions = tf.cast(predictions, tf.float64)

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None
        
    # 计算损失函数。
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 定义训练计数
    global_step = tf.Variable(0, trainable=False)

    # 定义衰减学习率
    learning_rate = tf.train.exponential_decay(0.01, global_step, 60, 0.95)

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, global_step, optimizer="Adagrad", learning_rate=learning_rate)
    return predictions, loss, train_op

def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    
    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算mse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    mse = mean_absolute_error(predictions, labels)
    print("Mean Square Error is: %f" % mse)

def output_to_csv():
    """output a csv file formed by predications"""

    submission = pd.read_csv(sample_submission_path+'sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = output_data
    print(submission.head())
    submission.to_csv(saving_path+'submission_LSTM.csv')

def test_model(sess):
    """output a csv file formed by predications"""

    with tf.variable_scope("model", reuse=True):
        X = tf.placeholder(tf.float64, [None, 1, TIMESTEPS])
        prediction, _, _ = lstm_model(X, [0.0], False)
    
    submission = pd.read_csv(sample_submission_path+'sample_submission.csv', index_col='seg_id', 
                             dtype={"time_to_failure": np.float64})
    for i, seg_id in enumerate(submission.index):
        seg = pd.read_csv(test_csv_path + seg_id + '.csv') 
        x = np.expand_dims(np.expand_dims(seg['acoustic_data'].values, 0), 0)   # why there needs to add dim? 
        submission.time_to_failure[i] = sess.run(prediction, {X:x})[0]

    print(submission.head())
    submission.to_csv(saving_path+'submission_LSTM.csv')

def get_raw_batch(data, min_index=0, max_index=None,
                    batch_size=32, n_steps=150, step_length=1000):
    """get a random batch. Note that input data is all of training data"""

    if max_index is None:
        max_index = len(data)-1

    # Pick indices of ending positions
    rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
    #rows = np.ones([batch_size], np.int)*150000
    # Initialize samples and targets
    raw_sample_data = []
    sample_labels = []   
    for j, row in enumerate(rows):
        x = data[(row - n_steps * step_length):row, 0]
        x = data_processing.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000)
        x = data_processing.denoise_signal(x, wavelet='haar', level=1)
        raw_sample_data.append(x)
        sample_labels.append(data[row - 1, 1])

    return raw_sample_data, sample_labels

def LSTM_test(watch_dog, preparation_threads):
    """
    test my LSTM model
    """

    # load all raw data
    float_data = pd.read_csv(context.traning_csv_path_name, 
                             dtype={"acoustic_data": np.int16, "time_to_failure": np.float64}).values

    #this part may be changed due to unlogical coding. 
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()

    #data = pd.read_csv(context.traning_csv_path_name).values
    #bqueue = data_processing.batch_queue_V3(context.on_cloud, get_raw_batch, (data))

    X = tf.placeholder(tf.float32, [None, 1, TIMESTEPS])
    y = tf.placeholder(tf.float64, [None, 1])

    # 定义模型，得到预测结果、损失函数，和训练操作。
    with tf.variable_scope("model"):
        model_output, loss, train_op = lstm_model(X, y, True)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 训练模型。
        max_unimp_i = 10000
        best_loss = 9999999
        unimp_i = 0
        while unimp_i < max_unimp_i:

            batch_X, batch_y = get_raw_batch(float_data)
            batch_X = np.expand_dims(batch_X, 1).astype(np.float64)    # why there needs to add dim? 
            batch_y = np.expand_dims(batch_y, 1)
            _, l, predictions = sess.run([train_op, loss, model_output], {X:batch_X, y:batch_y})
            if best_loss > l:
                unimp_i = 0
                best_loss = l
            else:
                unimp_i += 1

            #feed the dog and count iterations
            watch_dog.interval_feeding(100, "loss = %f"%(l))

        # 使用训练好的模型对测试数据进行预测。
        train_data = []
        train_labels = []
        for i in range(100):
            data_batch, label_batch = get_raw_batch(float_data)
            train_data.extend(data_batch)
            train_labels.extend(label_batch)
        run_eval(sess, train_data, train_labels)

        # evaluate test data and output a file 
        print("Evaluate test dat")
        test_model(sess)
