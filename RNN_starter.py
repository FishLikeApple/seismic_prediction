import data_processing
import seismic_prediction as context
'''
def RNN_starter_test(watch_dog, preparation_threads):
    # BASIC IDEA OF THE KERNEL

    # The data consists of a one dimensional time series x with 600 Mio data points. 
    # At test time, we will see a time series of length 150'000 to predict the next earthquake.
    # The idea of this kernel is to randomly sample chunks of length 150'000 from x, derive some
    # features and use them to update weights of a recurrent neural net with 150'000 / 1000 = 150
    # time steps. 

    import numpy as np 
    import pandas as pd
    import os
    """
    # Fix seeds
    from numpy.random import seed
    seed(639)
    from tensorflow import set_random_seed
    set_random_seed(5944)
    """
    # Import
    #float_data = pd.read_csv(context.traning_csv_path_name, dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
    float_data = np.zeros([150000])  #placehoder

    # Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
    # Can easily be extended. Expects a two dimensional array.
    def extract_features(z):
         return np.c_[z.mean(axis=1), 
                      z.min(axis=1),
                      z.max(axis=1),
                      z.std(axis=1)]

    # For a given ending position "last_index", we split the last 150'000 values 
    # of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
    # From each piece, a set features are extracted. This results in a feature matrix 
    # of dimension (150 time steps x features).  
    def create_X(x, last_index=None, n_steps=150, step_length=1000):
        if last_index == None:
            last_index=len(x)
       
        assert last_index - n_steps * step_length >= 0

        # Reshaping and approximate standardization with mean 5 and std 3.
        temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    
        # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
        # of the last 10 observations. 
        return np.c_[extract_features(temp),
                     extract_features(temp[:, -step_length // 10:]),
                     extract_features(temp[:, -step_length // 100:])]

    # Query "create_X" to figure out the number of features
    n_features = create_X(float_data[0:150000]).shape[1]
    print("Our RNN is based on %i features"% n_features)
    
    batch_size = 32

    # The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
    # the "time_to_failure" serves as target, while the features are created by the function "create_X".
    """
    def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
        if max_index is None:
            max_index = len(data) - 1
     
        while True:
            # Pick indices of ending positions
            rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
         
            # Initialize feature matrices and targets
            samples = np.zeros((batch_size, n_steps, n_features))
            targets = np.zeros(batch_size, )
        
            for j, row in enumerate(rows):
                samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
                targets[j] = data[row - 1, 1]
            yield samples, targets
    """

    #this part may be changed due to unlogical coding. 
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()
    #bqueue = data_processing.batch_queue(batch_size, context.csv_data_name, context.csv_time_name,
    #                                    context.on_cloud, context.traning_csv_piece_path, 4, create_X)

    def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
        """changed for my function"""
        
        while True:
            #feed the watch dog
            watch_dog.interval_feeding(1000)
            
            #get data to return        
            samples = []
            targets = []

            for i in range(batch_size):
                X_data, y_data = data_processing.get_data_from_pieces(context.traning_csv_piece_path, context.csv_data_name,
                                                                      context.csv_time_name)
                samples.append(create_X(X_data))
                targets.append(y_data[-1])

            yield np.array(samples), np.array(targets)

    # Position of second (of 16) earthquake. Used to have a clean split
    # between train and validation
    second_earthquake = 50085877
    #float_data[second_earthquake, 1]

    # Initialize generators
    train_gen = generator(float_data, batch_size=batch_size) # Use this for better score
    # train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)
    valid_gen = generator(float_data, batch_size=batch_size)

    # Define model
    from keras.models import Sequential
    from keras.layers import Dense, CuDNNGRU
    from keras.optimizers import adam
    from keras.callbacks import ModelCheckpoint

    cb = [ModelCheckpoint(context.saving_path+"model.hdf5", save_best_only=True, period=3)]

    model = Sequential()
    model.add(CuDNNGRU(48, input_shape=(None, n_features)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()

    # Compile and fit model
    model.compile(optimizer=adam(lr=0.0005), loss="mae")

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=1000,
                                  epochs=30,
                                  verbose=0,
                                  #callbacks=cb,
                                  validation_data=valid_gen,
                                  validation_steps=200)

    """
    # Visualize accuracies
    import matplotlib.pyplot as plt

    def perf_plot(history, what = 'loss'):
        x = history.history[what]
        val_x = history.history['val_' + what]
        epochs = np.asarray(history.epoch) + 1
    
        plt.plot(epochs, x, 'bo', label = "Training " + what)
        plt.plot(epochs, val_x, 'b', label = "Validation " + what)
        plt.title("Training and validation " + what)
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
        return None

    perf_plot(history)
    """
    # Load submission file
    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', 
                             index_col='seg_id', dtype={"time_to_failure": np.float64})

    # Load each test data, create the feature matrix, get numeric prediction
    for i, seg_id in enumerate(submission.index):
      #  print(i)
        seg = pd.read_csv(context.test_csv_path + seg_id + '.csv')
        x = seg['acoustic_data'].values
        submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

    submission.head()

    # Save
    submission.to_csv(context.saving_path+'submission_RNN.csv')
'''

def RNN_starter_test(watch_dog, preparation_threads, previous_model=None, 
                     original_generator=True, use_saved_model=False):
    # BASIC IDEA OF THE KERNEL

    # The data consists of a one dimensional time series x with 600 Mio data points. 
    # At test time, we will see a time series of length 150'000 to predict the next earthquake.
    # The idea of this kernel is to randomly sample chunks of length 150'000 from x, derive some
    # features and use them to update weights of a recurrent neural net with 150'000 / 1000 = 150
    # time steps. 

    import numpy as np 
    import pandas as pd
    import os
    from tqdm import tqdm
    import queue
    from sklearn.metrics import mean_absolute_error

    model_file_name = "RNN_starter_model.hdf5"

    """
    # Fix seeds
    from numpy.random import seed
    seed(639)
    from tensorflow import set_random_seed
    set_random_seed(5944)
    """

    # For a given ending position "last_index", we split the last 150'000 values 
    # of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
    # From each piece, a set features are extracted. This results in a feature matrix 
    # of dimension (150 time steps x features).  
    def create_X(x, last_index=None, n_steps=150, step_length=1000):
        if last_index == None:
            last_index=len(x)
       
        #with open(context.saving_path+'logs.txt', 'a') as f:
        #    print("last_index=%d"%(last_index), file=f)
        assert last_index - n_steps * step_length >= 0

        # Reshaping and approximate standardization with mean 5 and std 3.
        temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    
        # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
        # of the last 10 observations. 
        return np.c_[extract_features(temp),
                     extract_features(temp[:, -step_length // 10:]),
                     extract_features(temp[:, -step_length // 100:]),
                     np.c_(data_processing.get_frequency_feature(temp)[[10, 20, 40, 81]])]

    # Import
    watch_dog.feeding("loading main data file starts")
    float_data = pd.read_csv(context.traning_csv_path_name, 
                             dtype={"acoustic_data": np.int16, "time_to_failure": np.float64}).values
    watch_dog.feeding("main data file is loaded")

    batch_size = 32
    #bqueue = None

    #define the frist preprocessor
    def X_preprocessor(data, last_index, n_steps, step_length, batch_size=batch_size):
        """preprocessor for caching"""
        
        data_piece = data[(last_index - n_steps * step_length):last_index]
        if previous_model != None:  
            return [create_X(data, last_index, n_steps, step_length), 
                    previous_model.get_features_of_data_piece(data_piece)]
        else:
            return [create_X(data, last_index, n_steps, step_length), 
                    0]  # 0 for placeholding

    # this part may be changed due to unlogical coding. 
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()
    print("loading is completed")

    #bqueue = data_processing.batch_queue(batch_size, context.csv_data_name, context.csv_time_name,
    #                                    context.on_cloud, context.traning_csv_path_name, 4,
    #                                    create_X, second_preprocessor, use_lager_csv=True)
    #float_data = bqueue.lager_csv_data.values

    # Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
    # Can easily be extended. Expects a two dimensional array.
    
    def extract_features(z):
        """
        features = []
        for windows in [10, 100, 1000]:
            x_roll_std = x.rolling(windows).std().dropna().values
            x_roll_mean = x.rolling(windows).mean().dropna().values
        
            features.append(x_roll_std.mean())
            features.append(x_roll_std.std())
            features.append(x_roll_std.max())
            features.append(x_roll_std.min())
            features.append(np.quantile(x_roll_std, 0.01)
            features.append(np.quantile(x_roll_std, 0.05)
            features.append(np.quantile(x_roll_std, 0.95)
            features.append(np.quantile(x_roll_std, 0.99)
            features.append(np.mean(np.diff(x_roll_std))
            features.append(np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            features.append(np.abs(x_roll_std).max()
        
            features.loc[0, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
            features.loc[0, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
            features.loc[0, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
            features.loc[0, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
            features.loc[0, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            features.loc[0, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            features.loc[0, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            features.loc[0, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            features.loc[0, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
            features.loc[0, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            features.loc[0, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
        """
        return np.c_[z.mean(axis=1), 
                    z.min(axis=1),
                    z.max(axis=1),
                    z.std(axis=1)]

    # Query "create_X" to figure out the number of features
    n_features = create_X(float_data[0:150000]).shape[1]
    print("Our RNN is based on %i features"% n_features)
    
    # The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
    # the "time_to_failure" serves as target, while the features are created by the function "create_X".

    def get_raw_batch(data, min_index=0, max_index=None, 
                      batch_size=16, n_steps=150, step_length=1000):
        """get a random batch. Note that input data is all of training data
           Note that raw is not a right word to describe output of this function."""

        if max_index is None:
            #max_index = len(data) - 1
            max_index = len(data)

        # Pick indices of ending positions
        #rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
        rows = np.random.randint(min_index, max_index/(n_steps*step_length), size=batch_size) + 1
        rows = rows * n_steps * step_length
                    
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        data_list = []

        for j, row in enumerate(rows):
            data_list.append([X_preprocessor(data=data[:, 0], last_index=row, 
                                             n_steps=n_steps, step_length=step_length),   # may be modified
                             data[row - 1, 1]])

        #watch_dog.feeding(os.popen("free -h").read())  # debugging

        return data_list

    #define a segment preprocessor
    def second_preprocessor(samples, batch_size=batch_size):
        """samples are cached data"""

        assert (len(samples)%batch_size) == 0   # samples's shape must match batch_size

        #get all the input data of the 
        feature_input = []
        labels = []
        for sample in samples:
            feature_input.append(sample[0][1])
            labels.append(sample[1])

        #calculating residual labels
        if previous_model != None:
            output_labels = previous_model.predict_residuals(feature_input, labels, is_input_DF_features=True)
        else:
            output_labels = labels

        #form an output list
        output_data_batch = []
        output_label_batch = []
        output_list = []
        for i in range(len(samples)):
            output_data_batch.append(samples[i][0][0])
            output_label_batch.append(output_labels[i])
            if len(output_data_batch) == batch_size:
                output_list.append([output_data_batch, output_label_batch])
                output_data_batch = []
                output_label_batch = []

        assert len(output_list) == (len(samples)/batch_size)

        return output_list

    generators = data_processing.generator_generator(float_data, get_raw_batch, second_preprocessor, 
                                                     on_cloud=context.on_cloud, batch_size=batch_size)

    def modified_generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, 
                           step_length=1000, second_preprocessor=second_preprocessor):


        while True:
            #get a cached batch
            sample_batch = data_queue.get_batch()

            yield sample_batch[0], sample_batch[1]

    '''
    def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000, bqueue=bqueue):
        """changed for my function"""

        if max_index is None:
            max_index = len(data) - 1
        
        while True:
            #feed the watch dog
            watch_dog.interval_feeding(1000)
            
            # Pick indices of ending positions
            rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

            #get data to return        
            samples = []
            targets = []
            """
            X_data, y_data = bqueue.get_batch()
            for i in range(batch_size):
                samples.append(X_data[i])
                targets.append(y_data[len(y_data)-1])
            """
            for j, row in enumerate(rows):
                samples.append(create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length))
                targets.append(data[row - 1, 1])

            yield np.array(samples), np.array(targets)
    '''

    def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
        if max_index is None:
            max_index = len(data) + 1
     
        while True:
            # Pick indices of ending positions
            rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
            #rows = np.ones([batch_size], np.int)*n_steps*step_length*2   # for debugging
         
            # Initialize feature matrices and targets
            samples = np.zeros((batch_size, n_steps, n_features))
            targets = np.zeros(batch_size, )
        
            for j, row in enumerate(rows):
                samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
                targets[j] = data[row - 1, 1]
            yield samples, targets

    def test():  # debugging
        print(generators.generator().__next__())
        print(generator(float_data, batch_size=batch_size).__next__())
    #test()
    # Position of second (of 16) earthquake. Used to have a clean split
    # between train and validation
    second_earthquake = 50085877
    #float_data[second_earthquake, 1]


    # Initialize generators
    #train_gen = generators.generator() # Use this for better score
    train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake)
    valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)
    #valid_gen = generators.generator() #modified_generator(float_data, batch_size=batch_size, max_index=second_earthquake)

    if use_saved_model == True:
        print("use saved model")
    else:
        # Define model
        from keras.models import Sequential
        from keras.layers import Dense, CuDNNGRU, Reshape
        from keras.optimizers import adam
        from keras.callbacks import ModelCheckpoint
        from hyperas import distributions

        cb = [ModelCheckpoint("/"+model_file_name, save_best_only=True, period=3)]

        model = Sequential()
        model.add(CuDNNGRU({{distributions.choice([48, 96, 144])}}, input_shape=(None, n_features)))
        model.add(Dense({{distributions.choice([16, 32, 64])}}, activation={{distributions.choice(['relu', 'linear'])}}))
        model.add(Dense(1))
    
        model.summary()

        # Compile and fit model
        model.compile(optimizer=adam(lr=1/{{distributions.uniform(100, 10000)}}), loss="mae")

        watch_dog.feeding("training starts")
        history = model.fit_generator(train_gen,
                                      steps_per_epoch=1000,
                                      epochs={{distributions.randint(80)}}+20,
                                      verbose=0,
                                      callbacks=cb,
                                      validation_data=valid_gen,
                                      validation_steps=200)

        print(history.history['acc'])

        # Save. The model cannot be well saved on the cloud, so I use "copy" instead
        os.system("cp "+"/"+model_file_name+ " " + context.saving_path+model_file_name)

    #from keras.models import load_model
    #model = load_model(context.saving_path+model_file_name)

    # output result on 100*32 pieces of training data
    train_data = []
    train_labels = []
    for i in range(100):
        data_batch, label_batch = train_gen.__next__()
        train_data.extend(data_batch)
        train_labels.extend(label_batch)
    print("mean absolute error on training data:")
    print(mean_absolute_error(train_labels, model.predict(np.array(train_data))))

    # Load submission file
    submission = pd.read_csv(context.sample_submission_path+'sample_submission.csv', 
                             index_col='seg_id', dtype={"time_to_failure": np.float64})

    # Load each test data, create the feature matrix, get numeric prediction
    for i, seg_id in enumerate(tqdm(submission.index)):
      #  print(i)
        seg = pd.read_csv(context.test_csv_path + seg_id + '.csv')
        x = seg['acoustic_data'].values
        if previous_model != None:
            previous_model_prediction = previous_model.predict(previous_model.get_features_of_data_piece(x))
            submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0)) + previous_model_prediction
        else:
            submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

    submission.head()

    file_name = 'submission_RNN.csv'
    if previous_model != None:
        file_name = 'submission_RNN+gb.csv'
    elif original_generator == False:
        file_name = 'my_submission_RNN.csv'

    submission.to_csv(context.saving_path+file_name)