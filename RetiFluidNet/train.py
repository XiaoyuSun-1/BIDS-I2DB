# In[]
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Pick a GPU

from tqdm import tqdm
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
tf.random.set_seed(12345)
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import KFold
from DataReader import DataReader
# from models import Unet 
from model import RetiFluidNet
from losses import Losses, IntervalEvaluation
from results import Results
import glob

class Logger(tf.keras.callbacks.Callback):
    """
    Callback to log training progress.
    """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        # clear old log
        open(self.filepath, "w").close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filepath, "a") as f:
            f.write(f"Epoch {epoch+1}:\n")
            for name, value in logs.items():
                f.write(f"  {name}: {value:.4f}\n")
            f.write("\n")

# In[]: Setup
dataset_name = ['Spectralis','Cirrus']
output_dir = "_".join(dataset_name) + "_no_augs_e100" # Folder name for checkpoints and logs (e.g., "Spectralis_Cirrus_no_augs_e100")
os.makedirs(output_dir, exist_ok=True)

# Build the pooled case list 
data_path = []
for ds in dataset_name:
    root = f"RetouchData/{ds}/retouch_data"
    for case_folder in glob.glob(root + "/*"):
        data_path.append(case_folder)

print(f"Pooling {len(data_path)} cases from {dataset_name}")

# Utils & hyperparameters
data_reader = DataReader()
# unet = Unet(4, (256,256,1))  
loss_funcs = Losses()
my_results = Results()

train_flag = 1
do_continue = False
last_epoch = 0

SEED = 100
NUM_EPOCHS = 100
BATCH_SIZE = 4 #*nb_GPUs
BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE

# In[]: Main Loop
with open("retouch_splitted.json","r") as f:
    folds = json.load(f)
i = len(folds)
overall_results = []

# Learning rate decay schedule
def decay_schedule(epoch, lr):
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.8
    return lr    

for fold in folds:
    tf.random.set_seed(12345)
    print(f"Starting Fold number {i}")

    # Build case path datasets from JSON lists
    train_paths = tf.data.Dataset.from_tensor_slices(fold["train"]["images"])
    val_paths   = tf.data.Dataset.from_tensor_slices(fold["val"]["images"])
    # Map to (image, mask) using DataReader
    train_data, val_data = data_reader.get_data_for_train(train_paths, val_paths) 
        
    num_of_train_samples = len(train_data)
    num_of_val_samples   = len(val_data)
    for image, mask in val_data.skip(5).take(1):
        print("Image Shape : ", image.shape)
        print("Mask Shape  : ", mask.shape)

    # Data pipeline
    train_data = train_data.shuffle(buffer_size=BUFFER_SIZE, seed=SEED)\
                           .batch(BATCH_SIZE)\
                           .prefetch(buffer_size=AUTOTUNE)
    val_data   = val_data.batch(1).prefetch(buffer_size=AUTOTUNE)

    model = RetiFluidNet(num_class=4, input_shape=(256,256,1))()

    initial_learning_rate = 2e-4
    decay_steps = 10000
    decay_rate  = 0.98

    # Callbacks    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_schedule)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, f"model_{i}_checkpoint.hdf5"),
        save_best_only=True
    )
    log_path   = os.path.join(output_dir, f"log_fold_{i}.txt")
    logger = Logger(log_path)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(initial_learning_rate),
        loss=loss_funcs.training_loss,
        metrics=[loss_funcs.dice]
    )
    
    # Train or load model
    if train_flag:
        ival = IntervalEvaluation(validation_data=val_data)
        if do_continue:
            model = tf.keras.models.load_model(
                os.path.join(output_dir, f"model_{i}_epoch{last_epoch}.hdf5"),
                custom_objects={
                    'training_loss': loss_funcs.training_loss,
                    'dice_loss':     loss_funcs.training_loss,
                    'dice':          loss_funcs.dice
                }
            )
            print("Pre-trained model loaded.")

        # Fit
        history = model.fit(
            train_data,
            epochs=NUM_EPOCHS,
            callbacks=[ival, lr_scheduler, checkpoint_cb, logger]
        )

        # Save model and history
        final_path = os.path.join(output_dir, f"model_{i}_epoch{NUM_EPOCHS}.hdf5")
        model.save(final_path)
        np.save(os.path.join(output_dir, f"model_{i}_history.npy"), history.history)

    else:
        model = tf.keras.models.load_model(
            os.path.join(output_dir, f"model_{i}_epoch{NUM_EPOCHS}.hdf5"),
            custom_objects={
                'training_loss': loss_funcs.training_loss,
                'dice_loss':     loss_funcs.training_loss,
                'dice':          loss_funcs.dice
            }
        )

        # load history if needed
        hist_path = os.path.join(output_dir, f"model_{i}_history.npy")
        if os.path.exists(hist_path):
            History = np.load(hist_path, allow_pickle=True).item()
        else:
            print("No history file is found.")
 
        
    predictions = []
    for image, mask in tqdm(val_data):  
        # Only take connectivity channels 0:32
        temp = model.predict(image)[:, :, :, 0:32]
        predictions.append(temp)
    acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean = my_results.results_per_layer(predictions, val_data)
    overall_results.append([acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean])
    

        
    print('-'*50)
    print('Fold number {} finished'.format(i))
    print('-'*50)
    print('\n')
    print('\n')
    print('\n')
    print('\n')

 
    del model, train_data, val_data

    i -= 1     
    # break


# In[]:
my_results.print_overall_results(overall_results, dataset_name) 

print("SEED = %d\nNUM_EPOCHS = %d\nBATCH_SIZE = %d\nBUFFER_SIZE = %d"%(SEED,NUM_EPOCHS,BATCH_SIZE,BUFFER_SIZE))
print("initial_learning_rate = %.4f\ndecay_steps = %d\ndecay_rate = %0.2f"%(initial_learning_rate,decay_steps,decay_rate))
# In[]: END
