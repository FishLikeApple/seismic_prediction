import numpy as np
import csv
import pandas as pd
import os
import pandas as pd
import queue
import threading
import datetime
import time

csv.field_size_limit(1000000000)

def count_csv_rows(csv_file_name):
    """This function is to get the row number of a csv file."""

    with open(csv_file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            i += 1

    return i

def divide_lager_csv(csv_file_name, output_file_path, column1_name, column2_name, row_num_each_file=150000):
    """This function is to divide a lager csv file into multiple flies with a specified size."""

    with open(csv_file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        rows = []
        for row in reader:
            rows.append(row)
            if (i%row_num_each_file) == (row_num_each_file-1):
                with open(output_file_path+'piece'+str(int(i/row_num_each_file))+'.csv', 'w', newline='') as f:
                    csv_writer = csv.DictWriter(f, [column1_name, column2_name])
                    csv_writer.writeheader()
                    csv_writer.writerows(rows)
                rows = []
            i += 1
        if len(rows) > 0:
            with open(output_file_path+'piece'+str(int(i/row_num_each_file))+'.csv', 'w', newline='') as f:
                    csv_writer = csv.DictWriter(f, [column1_name, column2_name])
                    csv_writer.writeheader()
                    csv_writer.writerows(rows)

def get_file_paths_from_folder(folder, extensions=["csv"]):
    file_paths = []

    for extension in extensions:
        file_glob = glob.glob(folder+"/*."+extension)  #不分大小写
        file_paths.extend(file_glob)   #添加文件路径到file_paths

    return file_paths 

def get_data_from_csv(csv_file_name, column1_name, column2_name, start_index=0, len=None):
    """This function is to return sequence data from a csv file."""

    column1 = []
    column2 = []

    #train = pd.read_csv(csv_file_name)
    with open(csv_file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            column1.append(row[column1_name])
            column2.append(row[column2_name])

    return column1, column2

def get_data_from_pieces(csv_piece_path_or_data, column1_name, column2_name, start_point=None, 
                          data_len=150000, piece_len=150000, total_len=629145480, 
                          descending=True):    # in test
    """This function is to return a data piece from train csv pieces.
       Note that if csv_piece_path is not a path, it needs to be a list (array) of all data."""

    column1 = []
    column2 = []
    iterative = True
    if start_point == None:
        start_point = np.random.randint(0, total_len-data_len)
        iterative = False

    if type(csv_piece_path_or_data) != type("str"):
        return csv_piece_path_or_data[:data_len], csv_piece_path_or_data[:data_len]
    
    data = pd.read_csv(csv_piece_path_or_data+'piece'+str(int(start_point/piece_len))+'.csv')
    column1 = data[column1_name].values[(start_point%piece_len):]
    column2 = data[column2_name].values[(start_point%piece_len):]

    if column1.shape[0] <= data_len:
        data = pd.read_csv(csv_piece_path_or_data+'piece'+str(int(start_point/piece_len)+1)+'.csv')
        column1 = np.concatenate((column1, data[column1_name].values[:data_len-column1.shape[0]]),axis=0)
        column2 = np.concatenate((column2, data[column2_name].values[:data_len-column2.shape[0]]),axis=0)

    return column1[:data_len], column2[:data_len]

class batch_queue:
    """This class is for a batch queue with multiple threads for fast sampling."""

    def __init__(self, batch_size, column1_name, column2_name, on_cloud, csv_piece_path,
                 thread_num=4,  X_preprocessor=None, Y_preprocessor=None, use_lager_csv=False):
        """define a data queue and some threads, and start the threads.
           Note that X_preprocessor is a fuction with one input X, 
           but Y_preprocessor has two inputs X and y"""

        self.data_queue = queue.Queue(maxsize=thread_num*100)
        self.batch_size = batch_size
        self.csv_piece_path = csv_piece_path
        self.X_preprocessor = X_preprocessor
        self.Y_preprocessor = Y_preprocessor
        self.column1_name = column1_name
        self.column2_name = column2_name
        self.cache_thread = []
        thread_num += on_cloud*4

        self.lager_csv_data = None
        if use_lager_csv == True:
            self.lager_csv_data = pd.read_csv(csv_piece_path)

        for i in range(thread_num) :
            self.cache_thread.append(threading.Thread(target=self.batch_adder))
            self.cache_thread[i].daemon = True
            self.cache_thread[i].start()

    def add_batch(self):
        """add a batch to the queue"""

        batch_X = []
        batch_y = []

        for i in range(self.batch_size):
            if self.lager_csv_data == None:
                X, Y = get_data_from_pieces(self.csv_piece_path, self.column1_name, self.column2_name)
            else:
                X, Y = get_data_from_pieces(self.lager_csv_data, self.column1_name, self.column2_name)

            if self.X_preprocessor != None:
                X = self.X_preprocessor(X)
            y = Y[-1]
            if self.Y_preprocessor != None:
                y = self.Y_preprocessor(X, y)
            batch_X.append(X)
            batch_y.append(y)

        self.data_queue.put([batch_X, batch_y])

    def batch_adder(self):
        """always add batches if the queue is not full"""

        while True:
            self.add_batch()

    def get_batch(self):
        """get a batch"""
        return self.data_queue.get()

    def get_multi_batch_as_one(self, multiple):
        """get a customized large batch containing multiple batches"""

        batch_X = []
        batch_y = []

        for i in range(multiple):
            X, y = self.data_queue.get()
            batch_X.extend(X)
            batch_y.extend(y)

        return batch_X, batch_y

class batch_queue_V2:
    """This class is modified from the class above for combined models.
       Note that this class has many points (such as raw input, which can be just a data piece) to imporve."""

    def __init__(self, on_cloud, raw_batch_adder, adder_args=(), adder_num=1,
                 X_preprocessor=None, Y_preprocessor=None, preprocessor_num=1, queue_size_multiple=100):

        # cache raw data
        self.raw_data_queue = queue.Queue(maxsize=adder_num*queue_size_multiple)
        self.raw_batch_adder = raw_batch_adder
        self.adder_args = adder_args
        self.batch_adder_threads = []
        self.adder_num = adder_num + (on_cloud*4)  
        self.queue_size_multiple = queue_size_multiple
        for i in range(adder_num):
            self.batch_adder_threads.append(threading.Thread(target=self.batch_adder_thread))
            self.batch_adder_threads[i].daemon = True
            self.batch_adder_threads[i].start()

        # calculate and cache samples
        self.sample_queue = queue.Queue(maxsize=adder_num*queue_size_multiple)
        self.X_preprocessor = X_preprocessor
        self.Y_preprocessor = Y_preprocessor
        self.preprocessor_threads = []
        for i in range(preprocessor_num):
            self.preprocessor_threads.append(threading.Thread(target=self.preprocessor_thread))
            self.preprocessor_threads[i].daemon = True
            self.preprocessor_threads[i].start()

    def batch_adder_thread(self):

        while True:
            self.raw_data_queue.put(self.raw_batch_adder(adder_args[1]))

    def preprocessor_thread(self):

        X_buffer = []
        Y_buffer = []
        while True:
            data = self.raw_data_queue.get()
            X_buffer.extend(data[0])
            Y_buffer.extend(data[1])

            if len(X_buffer) == self.adder_num*self.queue_size_multiple:
                batch_size = len(data[0])
                if self.X_preprocessor == None:
                    sample_data = X_buffer
                else:
                    sample_data = self.X_preprocessor(X_buffer)
                if self.Y_preprocessor == None:
                    sample_labels = Y_buffer
                else:
                    sample_labels = self.Y_preprocessor(X_buffer, Y_buffer)
                for i in range(self.adder_num*self.queue_size_multiple):
                    self.sample_queue.put([sample_data[i*batch_size:(i+1)*batch_size],
                                          [sample_labels[i*batch_size:(i+1)*batch_size]]])
                X_buffer = []
                Y_buffer = []

    def get_batch(self):
        """get a sample batch"""

        return self.sample_queue.get()

class batch_queue_V3:
    """This class is modified from the class above for combined models.
       Note that adder_args is a parameter dictionary of RNN_starter's creat_X(),
       that is, adder_args[0] is whole raw data.

       raw_data_adder:a function used to add data pieces to a dada queue, 
                      it's input is a list of data pieces.

       segment_preprocessor:a dada preprocessor used to preprocess data in 
                            the dada queue and add a list of batches to a 
                            sample queue.
                            
       segment_input_len:input length of segment_preprocessor"""

    def __init__(self, on_cloud, raw_data_adder, adder_args=(), adder_num=4, 
                 segment_preprocessor=None, segment_input_len=32*15, segment_output_len=30, preprocessor_num=1, 
                 queue_size_multiple=100, watch_dog=None):

        self.watch_dog = watch_dog

        # cache raw data
        self.adder_num = adder_num + (on_cloud*4)
        self.raw_data_queue = queue.Queue(maxsize=self.adder_num*queue_size_multiple)
        self.raw_data_adder = raw_data_adder
        self.adder_args = adder_args
        self.data_adder_threads = []
        self.queue_size_multiple = queue_size_multiple
        for i in range(self.adder_num):
            self.data_adder_threads.append(threading.Thread(target=self.data_adder_thread))
            self.data_adder_threads[i].daemon = True
            self.data_adder_threads[i].start()

        # calculate and cache samples
        if segment_preprocessor == None:
            self.sample_queue = self.raw_data_queue
        else:
            self.sample_queue = queue.Queue(maxsize=segment_output_len)
            self.segment_preprocessor = segment_preprocessor
            self.segment_input_len = segment_input_len  
            self.preprocessor_threads = []
            for i in range(preprocessor_num):
                self.preprocessor_threads.append(threading.Thread(target=self.preprocessor_thread))
                self.preprocessor_threads[i].daemon = True
                self.preprocessor_threads[i].start()

    def data_adder_thread(self):
        """thread of the raw_batch_adder""" 

        while True:
            data_batch = self.raw_data_adder(*self.adder_args)
            for data in data_batch:
                self.raw_data_queue.put(data)

    def preprocessor_thread(self):
        """thread of the segment_preprocessor"""

        buffer = []

        while True:
            buffer.append(self.raw_data_queue.get())

            if len(buffer) == self.segment_input_len:
                batch_list = self.segment_preprocessor(buffer)

                for batch in batch_list:
                        self.sample_queue.put(batch)

                buffer.clear()

    def get_batch(self, is_array_output=True):
        """get a sample batch"""

        if is_array_output == False:
            return self.sample_queue.get()

        attribute_list = self.sample_queue.get()
        array_list = []
        for attributes in attribute_list:
            array_list.append(np.array(attributes))
        return array_list
            

    def check_and_print(self, text):
        """feed the wathcdog if it exists, otherwise just print the text."""

        if self.watch_dog != None:
            self.watch_dog.feeding(text)
        else:
            print(text)

class generator_generator:
    """This class is used for generating generators sharing the same data queue.
       Note that the parameter definition is from a certain context (RNN_starter)."""

    def __init__(self, data, raw_data_adder, segment_preprocessor, on_cloud, min_index=0, max_index=None, 
                    batch_size=16, n_steps=150, step_length=1000):

        if max_index is None:
            max_index = len(data) - 1
     
        self.data_queue = batch_queue_V3(on_cloud, raw_data_adder,
                                         (data, min_index, max_index, batch_size, n_steps, step_length), 
                                         segment_preprocessor=segment_preprocessor)    

    def generator(self):
        while True:
            #get a cached batch
            sample_batch = self.data_queue.get_batch()

            yield sample_batch[0], sample_batch[1]

def get_frequency_feature(input, window_size=None, mutil_dimension=False):

    if (window_size==None) or (window_size>=len(input)):
        return np.abs(np.fft.rfft(input))

    output = []
    for i in range(len(input)-window_size+1):
        result = np.abs(np.fft.rfft(input[i:i+window_size]))
        if mutil_dimension == False:
            output.append(np.mean(result))
        else:
            output.append(result)
    return np.array(output)

import zipfile
def zipDir(dirpath,outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

"""
def g():
    input = [np.sqrt(2)/2, 1, np.sqrt(2)/2, 0, np.sqrt(2)/-2, -1, np.sqrt(2)/-2, 0]
    output = np.fft.rfft(input)
    frequency = np.abs(output)
    print(frequency)
"""

class watchdog:
    """This is a watchdog class used for avoiding out-of-control on the cloud (and also logging)."""

    def __init__(self, food_path, on_cloud, log_path=None):
        self.food_path = food_path
        self.on_cloud = on_cloud
        self.log_path = log_path
        self.counting = 0

        if on_cloud==True:
            with open(food_path, "a") as f:
                print("enable watch dog")
            #thread = threading.Thread(target=self.watch_thread)
            #thread.daemon = True
            #thread.start()
        else:
            print("disable watch dog")

    def feeding(self, added_text=None, quiet=True):
        """stop the running code if food is not here"""
        if self.on_cloud == True:
            try:
                with open(self.food_path, "r") as f:
                    if quiet:
                        pass
                    elif self.log_path != None:
                        with open(self.log_path, "a") as f:
                            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=f)
                            if added_text != None:
                                print(added_text, file=f)
                            print("", file=f)
                    else:
                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        if added_text != None:
                            print(added_text)
                        print("")
            except:
                os._exit(1)

    def interval_feeding(self, interval, added_text=None):
        """feed the dog in intervals"""

        self.counting += 1
        if self.counting % interval == 0:
            self.feeding(added_text)

    def watch_thread(self):
        """automatically watch"""

        while True:
            time.sleep(5)
            self.feeding("auto feeding")

def install_package_offline(pack_path):
    """creat a thread to install a needed package offline"""

    thread = threading.Thread(target=os.system, args=("pip install "+pack_path,))
    thread.daemon = True
    thread.start()
    return thread

def package_installer(package_names):
    """install needed packages"""

    for package_name in package_names:
        os.popen("pip install --upgrade "+package_name).read()

def install_packages(package_names):
    """creat a thread to install needed packages"""

    thread = threading.Thread(target=package_installer, args=(package_names,))
    thread.daemon = True
    thread.start()
    return thread

def unzip_in_same_folder(file_path):
    os.popen("unzip -d "+os.path.splitext(file_path)[0]+" "+file_path).read()

def load_flie(input_path_name, output_path, unzip=False):
    """load training and test data"""

    os.popen("cp "+input_path_name+" "+output_path).read()
    if unzip == True:
        thread = threading.Thread(target=unzip_in_same_folder, args=(output_path,))
        thread.daemon = True
        thread.start()
        return thread

    return None

import sys
import importlib
def satellite_thread(watch_dog, subwork_names=[], outout_file_path_name=None, second_delay=4):
    """set this before running on the cloud"""

    if len(subwork_names) != 0:
        subwork_list = list(np.zeros(len(subwork_names)))
    else:
        subwork_list = []
    while True:
        try:
            if outout_file_path_name != None:
                output = open(outout_file_path_name, 'a')
                sys.stdout = output
                time.sleep(second_delay)
                sys.stdout.close()
                sys.stderr.flush()
            else:
                time.sleep(second_delay)
                sys.stdout.flush()
                sys.stderr.flush()
            watch_dog.feeding()

            # try to run subwork
            for i in range(len(subwork_names)):
                try:
                    subwork = importlib.import_module(name)
                    if subwork_list[i] != 0:
                        subwork_list[i] = threading.Thread(target=subwork.main, args=(0,))
                        subwork_list[i].daemon = True
                        subwork_list[i].start()

                except:
                    subwork_list[i] = 0
        except:
            os._exit(None)

def on_cloud_print_setting(saving_path, on_cloud, subwork_names=[], outout_file_path_name=None):
    # creat a satellite thread

    watch_dog = watchdog(saving_path+"food", on_cloud)
    thread = threading.Thread(target=satellite_thread, 
                              args=(watch_dog, subwork_names, outout_file_path_name))
    thread.daemon = True
    thread.start()

    return thread

class batch_generator_V2:
    """this class is used to generate a data batch from all raw data"""

    def __init__(self, raw_csv_path_name):
        self.float_data = pd.read_csv(raw_csv_path_name, 
                                      dtype={"acoustic_data": np.int16, "time_to_failure": np.float64}).values

    def get_raw_batch(self, data=None, min_index=0, max_index=None,
                      batch_size=32, n_steps=150, step_length=1000):
        """get a random batch. Note that input data is all of training data"""

        if data is None:
            data = self.float_data

        if max_index is None:
            max_index = len(data)-1

        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
        #rows = np.ones([batch_size], np.int)*150000
        # Initialize samples and targets
        raw_sample_data = []
        sample_labels = []   
        for j, row in enumerate(rows):
            raw_sample_data.append(data[(row - n_steps * step_length):row, 0])
            sample_labels.append(data[row - 1, 1])

        return raw_sample_data, sample_labels

import glob
def submission_blending(csv_path, output_file_path_name):
    """output a combined submission file of multiple submission files in csv_path"""

    # load csv files
    csv_list = []
    file_glob = glob.glob(csv_path+"/*.csv")  #不分大小写
    for file_path in file_glob:
        csv_list.append(pd.read_csv(file_path, index_col='seg_id', dtype={"time_to_failure": np.float64}))

    csv_list[0]['time_to_failure'] = csv_list[0]['time_to_failure']/len(csv_list)
    for csv_ in csv_list[1:]:
        csv_list[0]['time_to_failure'] += csv_['time_to_failure']/len(csv_list)

    csv_list[0].to_csv(output_file_path_name)

def weighted_submission_blending(csv_path_name_list, weight_list, output_file_path_name):
    """output a combined submission file of multiple submission files in csv_path"""

    # load csv files
    csv_list = []
    for file_path in csv_path_name_list:
        csv_list.append(pd.read_csv(file_path, index_col='seg_id', dtype={"time_to_failure": np.float64}))

    # weight output
    weight_count = np.sum(weight_list)
    csv_list[0]['time_to_failure'] = csv_list[0]['time_to_failure']*weight_list[0]/weight_count
    for i in range(len(csv_list[1:])):
        csv_list[0]['time_to_failure'] += csv_list[i+1]['time_to_failure']*weight_list[i+1]/weight_count

    csv_list[0].to_csv(output_file_path_name)

# denoising
from numpy.fft import *
import pandas as pd
import pywt 
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter, deconvolve
import warnings
warnings.filterwarnings('ignore')

SIGNAL_LEN = 150000
SAMPLE_RATE = 4000

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, SAMPLE_RATE=SAMPLE_RATE):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

# function of downloading image
import urllib
from PIL import Image
from io import BytesIO

def DownloadImage(key_url, out_dir):
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        response = urllib.request.urlopen(url)
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(response.read()))
    except:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return

# feature generator from masterpiece1
def features(x, y, seg_id):
    with open(context.saving_path+'logs.txt', 'a') as f:
        print("start to get features at " + str(seg_id), file=f)
    feature_dict = dict()
    feature_dict['target'] = y
    feature_dict['seg_id'] = seg_id

    # create features here
    # numpy

    # lists with parameters to iterate over them
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    hann_windows = [50, 150, 1500, 15000]
    spans = [300, 3000, 30000, 50000]
    windows = [10, 50, 100, 500, 1000, 10000]
    borders = list(range(-4000, 4001, 1000))
    peaks = [10, 20, 50, 100]
    coefs = [1, 5, 10, 50, 100]
    lags = [10, 100, 1000, 10000]
    autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]

    # basic stats
    feature_dict['mean'] = x.mean()
    feature_dict['std'] = x.std()
    feature_dict['max'] = x.max()
    feature_dict['min'] = x.min()

    # basic stats on absolute values
    feature_dict['mean_change_abs'] = np.mean(np.diff(x))
    feature_dict['abs_max'] = np.abs(x).max()
    feature_dict['abs_mean'] = np.abs(x).mean()
    feature_dict['abs_std'] = np.abs(x).std()

    # geometric and harminic means
    feature_dict['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
    feature_dict['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]])) 

    # k-statistic and moments
    for i in range(1, 5):
        feature_dict['kstat_' + str(i)] = stats.kstat(x, i)
        feature_dict['moment_' + str(i)] = stats.moment(x, i)

    for i in [1, 2]:
        feature_dict['kstatvar_' +str(i)] = stats.kstatvar(x, i)

    # aggregations on various slices of data
    for agg_type, slice_length, direction in product(['std', 'min', 'max', 'mean'], [1000, 10000, 50000], ['first', 'last']):
        if direction == 'first':
            feature_dict[agg_type + '_' + direction + '_' + str(slice_length)] = x[:slice_length].agg(agg_type)
        elif direction == 'last':
            feature_dict[agg_type + '_' + direction + '_' + str(slice_length)] = x[-slice_length:].agg(agg_type)

    feature_dict['max_to_min'] = x.max() / np.abs(x.min())
    feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
    feature_dict['count_big'] = len(x[np.abs(x) > 500])
    feature_dict['sum'] = x.sum()

    feature_dict['mean_change_rate'] = calc_change_rate(x)
    # calc_change_rate on slices of data
    for slice_length, direction in product([1000, 10000, 50000], ['first', 'last']):
    #for slice_length, direction in product([50000], ['first', 'last']):
        if direction == 'first':
            feature_dict['mean_change_rate_' + direction+ '_' + str(slice_length)] = calc_change_rate(x[:slice_length])
        elif direction == 'last':
            feature_dict['mean_change_rate_' + direction+ '_' + str(slice_length)] = calc_change_rate(x[-slice_length:])

    # percentiles on original and absolute values
    for p in percentiles:
        feature_dict['percentile_' + str(p)] = np.percentile(x, p)
        feature_dict['abs_percentile_' + str(p)] = np.percentile(np.abs(x), p)

    feature_dict['trend'] = add_trend_feature(x)
    feature_dict['abs_trend'] = add_trend_feature(x, abs_values=True)

    feature_dict['mad'] = x.mad()
    feature_dict['kurt'] = x.kurtosis()
    feature_dict['skew'] = x.skew()
    feature_dict['med'] = x.median()

    feature_dict['Hilbert_mean'] = np.abs(hilbert(x)).mean()

    for hw in hann_windows:
        feature_dict['Hann_window_mean_' + str(hw)] = (convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

    feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    feature_dict['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    feature_dict['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    feature_dict['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    feature_dict['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    feature_dict['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    feature_dict['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    # exponential rolling statistics
    ewma = pd.Series.ewm
    for s in spans:
        feature_dict['exp_Moving_average_'+ str(s) + '_mean'] = (ewma(x, span=s).mean(skipna=True)).mean(skipna=True)
        feature_dict['exp_Moving_average_'+ str(s) + '_std'] = (ewma(x, span=s).mean(skipna=True)).std(skipna=True)
        feature_dict['exp_Moving_std_' + str(s) + '_mean'] = (ewma(x, span=s).std(skipna=True)).mean(skipna=True)
        feature_dict['exp_Moving_std_' + str(s) + '_std'] = (ewma(x, span=s).std(skipna=True)).std(skipna=True)

    feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    feature_dict['iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
    feature_dict['ave10'] = stats.trim_mean(x, 0.1)

    # tfresh features take too long to calculate, so I comment them for now

    feature_dict['abs_energy'] = feature_calculators.abs_energy(x)
    feature_dict['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
    feature_dict['count_above_mean'] = feature_calculators.count_above_mean(x)
    feature_dict['count_below_mean'] = feature_calculators.count_below_mean(x)
    feature_dict['mean_abs_change'] = feature_calculators.mean_abs_change(x)
    feature_dict['mean_change'] = feature_calculators.mean_change(x)
    feature_dict['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
    feature_dict['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
    feature_dict['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

    for i, j in zip(borders, borders[1:]):
        feature_dict['range_'+ str(i) +'_' + str(j)] = feature_calculators.range_count(x, i, j)

    feature_dict['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
    feature_dict['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
    feature_dict['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
    feature_dict['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
    feature_dict['last_loc_max'] = feature_calculators.last_location_of_maximum(x)

    for lag in lags:
        feature_dict['time_rev_asym_stat_'+str(lag)] = feature_calculators.time_reversal_asymmetry_statistic(x, lag)
    for autocorr_lag in autocorr_lags:
        feature_dict['autocorrelation_' + str(autocorr_lag)] = feature_calculators.autocorrelation(x, autocorr_lag)
        feature_dict['c3_' + str(autocorr_lag)] = feature_calculators.c3(x, autocorr_lag)

    for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
        feature_dict['fft_'+str(coeff)+'_'+attr] = list(feature_calculators.fft_coefficient(x, [{'coeff': coeff, 'attr': attr}]))[0][1]

    feature_dict['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
    feature_dict['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
    feature_dict['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
    feature_dict['cid_ce_1'] = feature_calculators.cid_ce(x, 1)

    for p in percentiles:
        feature_dict['binned_entropy_'+str(p)] = feature_calculators.binned_entropy(x, p)

    feature_dict['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

    for peak in peaks:
        feature_dict['num_peaks_' + str(peak)] = feature_calculators.number_peaks(x, peak)

    for c in coefs:
        feature_dict['spkt_welch_density_' + str(c)] = list(feature_calculators.spkt_welch_density(x, [{'coeff': c}]))[0][1]
        feature_dict['time_rev_asym_stat_' + str(c)] = feature_calculators.time_reversal_asymmetry_statistic(x, c)  

    # statistics on rolling windows of various sizes
        
    for w in windows:
        x_roll_std = x.rolling(w).std().dropna().values
        x_roll_mean = x.rolling(w).mean().dropna().values

        feature_dict['ave_roll_std_' + str(w)] = x_roll_std.mean()
        feature_dict['std_roll_std_' + str(w)] = x_roll_std.std()
        feature_dict['maxx_roll_std_' + str(w)] = x_roll_std.max()
        feature_dict['min_roll_std_' + str(w)] = x_roll_std.min()

        for p in percentiles:
            feature_dict['percentile_roll_std_' + str(p)] = np.percentile(x_roll_std, p)

        feature_dict['av_change_abs_roll_std_' + str(w)] = np.mean(np.diff(x_roll_std))
        feature_dict['av_change_rate_roll_std_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        feature_dict['abs_max_roll_std_' + str(w)] = np.abs(x_roll_std).max()

        feature_dict['ave_roll_mean_' + str(w)] = x_roll_mean.mean()
        feature_dict['std_roll_mean_' + str(w)] = x_roll_mean.std()
        feature_dict['max_roll_mean_' + str(w)] = x_roll_mean.max()
        feature_dict['min_roll_mean_' + str(w)] = x_roll_mean.min()

        for p in percentiles:
            feature_dict['percentile_roll_mean_' + str(p)] = np.percentile(x_roll_mean, p)

        feature_dict['av_change_abs_roll_mean_' + str(w)] = np.mean(np.diff(x_roll_mean))
        feature_dict['av_change_rate_roll_mean_' + str(w)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        feature_dict['abs_max_roll_mean_' + str(w)] = np.abs(x_roll_mean).max()       
        
    return feature_dict

# as above
def get_features(x, y, seg_id):

    x = pd.Series(x)

    x_denoised = data_processing.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000)
    x_denoised = data_processing.denoise_signal(x_denoised, wavelet='haar', level=1)
    x_denoised = pd.Series(x_denoised)
    
    zc = np.fft.fft(x)
    realFFT = pd.Series(np.real(zc))
    imagFFT = pd.Series(np.imag(zc))
    zc_denoised = np.fft.fft(x_denoised)
    realFFT_denoised = pd.Series(np.real(zc_denoised))
    imagFFT_denoised = pd.Series(np.imag(zc_denoised))
        
    main_dict = self.features(x, y, seg_id)
    r_dict = self.features(realFFT, y, seg_id)
    i_dict = self.features(imagFFT, y, seg_id)
    main_dict_denoised = self.features(x_denoised, y, seg_id)
    r_dict_denoised = self.features(realFFT_denoised, y, seg_id)
    i_dict_denoised = self.features(imagFFT_denoised, y, seg_id)
        
    for k, v in r_dict.items():
        if k not in ['target', 'seg_id']:
            main_dict['fftr_' + str(k)] = v
                
    for k, v in i_dict.items():
        if k not in ['target', 'seg_id']:
            main_dict['ffti_' + str(k)] = v

    for k, v in main_dict_denoised.items():
        if k not in ['target', 'seg_id']:
            main_dict['denoised_' + str(k)] = v

    for k, v in r_dict_denoised.items():
        if k not in ['target', 'seg_id']:
            main_dict['denoised_fftr_' + str(k)] = v
                
    for k, v in i_dict_denoised.items():
        if k not in ['target', 'seg_id']:
            main_dict['denoised_ffti_' + str(k)] = v
        
    return main_dict

def delete_csv_space_bar(input_csv_path_name, output_csv_path_name):
    """delete all space bars in the input csv"""

    csv_file = pd.read_csv(input_csv_path_name)
    print(csv_file.head())
    csv_file.dropna() 
    print(csv_file.head())
    csv_file.to_csv(output_csv_path_name, index=False)

def feature_filter(features, feature_importance , threshold=30):
    """filter features by importance"""

    returned_features = pd.DataFrame(index=features.index)
    for i in range(len(features.columns)):
        right_lines = feature_importance[feature_importance["feature"]==features.columns[i]]
        if np.mean(right_lines.loc[:,'importance']) > threshold:
            a = features[features.columns[i]]
            returned_features = pd.concat([returned_features, features[features.columns[i]]], axis=1)
    
    return returned_features 

on_cloud = False
try:
    import on_PC
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib import cm
    import matplotlib as mpl
    class feature_saver:
        def __init__(self, output_path, cmap='viridis', lu=256):
            self.output_path = output_path
            self.viridis = cm.get_cmap(cmap, lu)

        def save_feature_as_2Dimage(self, data, min_threshold, max_threshold, label):
            """
            This function is to show or save a colormap.
            """
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))  #Originally there has a parameter constrained_layout 
                                                           #in the example, but it does work locally.
            psm = ax.pcolormesh(data, cmap=self.viridis, rasterized=True, vmin=min_threshold, vmax=max_threshold)
            fig.colorbar(psm, ax=ax)
            #plt.show()
            plt.savefig(self.output_path+str(label)+".jpg")
            plt.clf()
    
    import random
    from example.commons import Faker
    from pyecharts import options as opts
    from pyecharts.charts import Bar3D
    from pyecharts_snapshot.main import make_a_snapshot
    class feature_saver_3D:
        """This is a class for saving features as a 3D colormap"""

        def __init__(self, output_path, light_source=[0, 0], rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False):
            self.output_path = output_path
            self.ls = LightSource(light_source[0], light_source[0])
            self.rstride = rstride 
            self.cstride = cstride
            self.linewidth = linewidth 
            self.antialiased = False 
            self.shade = False

        def save_feature_as_3Dimage(self, data_array, output_file_name):
            """
            This function is to show or save a 3D colormap.
            """

            column_len = int(np.sqrt(data_array.shape[0]))

            y = column_len
            x = data_array.shape[0]//y
            if (data_array.shape[0]%y) != 0:
                x += 1
            x, y = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))
            
            pad = np.zeros([x.shape[0]*x.shape[1]-data_array.shape[0]])
            #random.seed(999)
            #random.shuffle(data_array)
            z = np.concatenate([data_array, pad])
            z = np.reshape(z, x.shape)

            x = np.reshape(x, [-1, 1])
            y = np.reshape(y, [-1, 1])
            z = np.reshape(z, [-1, 1])
            data_xyz = np.concatenate([y, x, z], 1)

            bar3d = Bar3D()
            bar3d.add('', data_xyz, xaxis3d_opts=opts.Axis3DOpts(type_="value", min_=np.min(x), max_=np.max(x)),
                                    yaxis3d_opts=opts.Axis3DOpts(type_="value", min_=np.min(y), max_=np.max(y)),
                                    zaxis3d_opts=opts.Axis3DOpts(type_="value", min_=1))
            bar3d.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=1),
                                  title_opts=opts.TitleOpts(title="Bar3D-基本示例"))

            bar3d.render()
            make_a_snapshot('render.html', self.output_path+output_file_name)
            """
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            # To use a custom hillshading mode, override the built-in shading and pass
            # in the rgb colors of the shaded surface calculated from "shade"
            #norm = mpl.colors.Normalize(vmin=min_threshold, vmax=max_threshold)
            #rgb = self.ls.shade(z, cmap=cm.get_cmap('gist_earth'), vert_exag=0.1, blend_mode='soft', norm=norm)
            
            
            ax.plot_surface(x, y, z, facecolors=rgb, rstride=self.rstride, cstride=self.cstride,
                            linewidth=self.linewidth, antialiased=False, shade=False)
            
            color_list = []
            for i in range((x.shape[0]*x.shape[1]//5)+1):
                    color_list.extend(['r', 'g', 'm', 'y', 'k'])
            color_list = color_list[:x.shape[0]*x.shape[1]]
               
            ax.bar3d(x.ravel(), y.ravel(), 0, 0.5, 0.5, z.ravel(), zsort='average')#,  color=color_list)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            '''
            fig = plt.figure()
            ax = Axes3D(fig)
            
            ax.plot_surface(x, y, z, rstride=0.01, cstride=0.01, cmap=cm.viridis)
            '''
            ax.view_init(elev=60, azim=240)
            fig.savefig(self.output_path+output_file_name, dpi=200)
            clip_image(self.output_path+output_file_name, self.output_path+output_file_name, 
                       [320, 150], [960, 790])
            plt.show()
            fig.clf()
            """
    

    def save_feature_as_images():
        """for testing"""

        saver=feature_saver("D:/Backup/Documents/Visual Studio 2015/LANL-Earthquake-Prediction/master_data_graphs/")
        for i in range(10000):
            a, b = get_data_from_pieces(traning_csv_piece_path, csv_data_name, csv_time_name)
            add = np.zeros([(388*388)-150000], int)
            a = np.concatenate((a, add), axis=0)
            a = a.reshape([388, 388])
            saver.save_feature_as_image(a, -50, 50, b[-1])

    def save_features_as_3Dimages(feature_csv, feature_importance, output_path, importance_threshold=2, 
                                  data_offset=1, data_multiple=1/100):
        """get features for DCNN"""

        feature_csv = feature_filter(feature_csv, feature_importance, importance_threshold)
        saver=feature_saver_3D(output_path)
        for i in range(len(feature_csv)):
            data = (feature_csv.iloc[i,:]+data_offset) * data_multiple
            saver.save_feature_as_3Dimage(data, str(i)+'.jpg')

    from skimage import io
    def clip_image(input_path_name, output_path_name, output_first_corner, output_last_corner):
        """cut an image with the same center and a specified size ratio"""
        img=io.imread(input_path_name)
        img = img[output_first_corner[1]:output_last_corner[1], output_first_corner[0]:output_last_corner[0], :]
        io.imsave(output_path_name, img)

except:
    on_cloud = True
