"""

"""

on_cloud = False
traning_csv_path_name = "D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/train.csv"
traning_csv_piece_path = "D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/train_pieces/"
sample_submission_path = "D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/"
test_csv_path = "D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/test/"
saving_path = "D:/Backup/Documents/Visual Studio 2015/Projects/seismic_prediction_v/seismic_prediction/outputs/"
csv_data_name = "acoustic_data"
csv_time_name = "time_to_failure"
ckpt_file = "D:/Backup/Documents/Visual Studio 2015/inception_v4.ckpt"

try:
    import on_PC
except:
    on_cloud = True
    traning_csv_path_name = "/cos_person/275/1745/eq_files/train.csv"
    traning_csv_piece_path = "/train_pieces/"
    sample_submission_path = "/cos_person/275/1745/eq_files/"
    test_csv_path = "/test/"
    saving_path = "/cos_person/275/1745/eq_test/outputs/"
    ckpt_file = "/cos_person/275/1745/inception_v4.ckpt"

def wait_for_presetting(preparation_threads):
    """wait for presetting threads on cloud"""

    print("wait for presetting")
    for preparation_thread in preparation_threads:
        if preparation_thread != None:
            preparation_thread.join()

def main(_):

    print("start")

    #load the basic environment
    import os
    if on_cloud == True:
        os.popen("pip install --upgrade "+"numpy").read()
        os.popen("pip install --upgrade "+"PyWavelets").read()
        os.popen("pip install --upgrade "+"statsmodels").read()
    print("basic environment is loaded")
    import data_processing

    #set log output
    if on_cloud != 99:
        data_processing.on_cloud_print_setting(saving_path, on_cloud)#saving_path+'stdlog.txt')

    #set tasks which should be completed before training, this part may be changed due to poor hierarchy
    preparation_threads = []
    if on_cloud == True:
        # use two lines of code below to load and use csv pieces
        #thread = data_processing.load_flie("/cos_person/275/1745/eq_files/train_pieces.zip", "/train_pieces.zip", True)
        #preparation_threads.append(thread)
        #thread = data_processing.load_flie("/cos_person/275/1745/eq_files/test.zip", "/test.zip", True)
        #preparation_threads.append(thread)

        package_names = ["keras", "lightgbm", "xgboost", "catboost", "tqdm", "hyperopt", "tsfresh", 
                         "joblib", "hyperas", "gplearn"]
        data_processing.package_installer(package_names)

    import scipy
    import numpy as np

    from sklearn.metrics import mean_absolute_error
    import LSTM
    import RNN_starter
    import masterpiece1
    import masterpiece2
    import genetic_program
    import hyperopt_test
    import NN
    import threading
    import pandas as pd

    #define a watch dog
    watch_dog = data_processing.watchdog(saving_path+"food", on_cloud)#, saving_path+"logs.txt")
    # load the previous model for gb
    previous_model = None

    if on_cloud:
        #thread1 = threading.Thread(target=hyperopt_test.hyperopt_test, 
        #                           args=(preparation_threads, 2))
        #thread1.daemon=True
        #thread1.start()
        #wait_for_presetting(preparation_threads)
        #masterpiece2.make_on_overlap_data_csv(output_file_name='no_lap_train')
        #masterpiece2.scale_one_field(saving_path+'no_lap_train_X.csv', saving_path+'scaled_no_lap_train_X.csv')
        #hyperopt_test.hyperopt_test(preparation_threads, 'gp')
        #hyperopt_test.hyperopt_test(preparation_threads, 2)
        
        masterpiece2.master_test(preparation_threads, 'xgb', 125)
        masterpiece2.master_test(preparation_threads, 'xgb', 150)
        masterpiece2.master_test(preparation_threads, 'xgb', 175)
        masterpiece2.master_test(preparation_threads, 'xgb', 200)
        masterpiece2.master_test(preparation_threads, 'xgb', 225)
        #masterpiece2.master_test(preparation_threads, 'cat', 1000)
        '''
        evas = np.array([2396, 2173, 2629, 2340, 2715, 2525, 2538, 2600])
        n_eva = 15
        n_fold = 8
        evas = (evas*n_eva*n_fold/np.sum(evas)+0.5).astype(np.int)
        print(evas)
        masterpiece2.master_test(preparation_threads, 'xgb', evas, n_fold, 'ave_n_est'+str(n_eva))

        evas = np.array([2396, 2173, 2629, 2340, 2715, 2525, 2538, 2600])
        n_eva = 17
        n_fold = 8
        evas = (evas*n_eva*n_fold/np.sum(evas)+0.5).astype(np.int)
        print(evas)
        masterpiece2.master_test(preparation_threads, 'xgb', evas, n_fold, 'ave_n_est'+str(n_eva))
        '''
        #thread1.join()
        #masterpiece2.master_test(preparation_threads)
        #LSTM.LSTM_test(watch_dog, preparation_threads)
        #NN.NN_model(preparation_threads)
        #previous_model = masterpiece1.combined_prediction_model(saving_path, watch_dog=watch_dog)
        #previous_model.train_model_Z('stacking_model1.csv', preparation_threads)
        #RNN_starter.RNN_starter_test(watch_dog, preparation_threads, previous_model)
        #LSTM.LSTM_test(watch_dog, preparation_threads)
        #previous_model = masterpiece1.combined_prediction_model(saving_path)
        #genetic_program.genetic_program_test('genetic_program_sub.csv', preparation_threads)
        '''
        masterpiece1.master_test(preparation_threads)
        previous_model = masterpiece1.combined_prediction_model(saving_path)
        previous_model.test('combined_model_test1.csv')
        previous_model = masterpiece1.combined_prediction_model(saving_path)
        previous_model.test('combined_model_test2.csv')
        '''
    else:
        #NN.NN_model(preparation_threads)
        #masterpiece2.master_test(preparation_threads, 'xgb', 30)
        #hyperopt_test.hyperopt_test(preparation_threads,'gp')
        #data_processing.clip_image(sample_submission_path+'CNN_input/0.jpg', sample_submission_path+'CNN_input/00.jpg', 
        #                           [320, 120], [960, 760])
        '''
        feature_csv = pd.read_csv(saving_path+'scaled_train_X.csv')
        feature_importance_csv = pd.read_csv(saving_path+'feature_importance.csv')
        data_processing.save_features_as_3Dimages(feature_csv, feature_importance_csv, sample_submission_path+'CNN_input/')
        '''
        # load csv files
        csv_name_list = []
        weight_list = [1, 1, 1]                                    
        file_glob = data_processing.glob.glob("D:\Backup\Documents\Visual Studio 2015\Projects\seismic_prediction_v\seismic_prediction\outputs\submission_collection"+"/*.csv")  #不分大小写
        for file_path in file_glob:
            csv_name_list.append(file_path)
        data_processing.weighted_submission_blending(csv_name_list, weight_list, saving_path+"f_weighted_sub_collection.csv")
        
        #masterpiece2.make_new_data_csv(27, 5000, 8400, 'new_samples_8401-13400', 850)
        #wait_for_presetting(preparation_threads)
        #masterpiece2.make_on_overlap_data_csv(output_file_name='no_lap_train')
        #masterpiece2.master_test(preparation_threads)
        #hyperopt_test.hyperopt_test(preparation_threads, 3)
        #data_processing.submission_blending("D:\Backup\Documents\Visual Studio 2015\Projects\seismic_prediction_v\seismic_prediction\outputs\submission_collection",
        #                                    saving_path+"sub_collection.csv")
        #masterpiece1.master_test(preparation_threads)
        #previous_model = masterpiece1.combined_prediction_model(saving_path)
        #previous_model.train_model_Z('stacking_model1.csv')
        #genetic_program.genetic_program_test('genetic_program_sub.csv')
        #previous_model = masterpiece1.combined_prediction_model(saving_path)
        #RNN_starter.RNN_starter_test(watch_dog, preparation_threads, previous_model)
        #previous_model = masterpiece1.combined_prediction_model(saving_path)
        #previous_model.test("combined_model_test2.csv")
        #masterpiece1.master_test(preparation_threads)
        #RNN_starter.RNN_starter_test(watch_dog, preparation_threads, previous_model)
    #RNN_starter.RNN_starter_test(watch_dog, preparation_threads, previous_model,True)
    #RNN_starter.RNN_starter_test(watch_dog, preparation_threads, previous_model,False)
 
    #LSTM.LSTM_test(watch_dog, preparation_threads)
    a = 0

if on_cloud == False and __name__ == '__main__':
    main(0)