# In[]
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
tf.random.set_seed(12345)
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from DataReader import DataReader
# from models import Unet   # modified: comment out 
from model import RetiFluidNet
from losses import Losses, IntervalEvaluation
from results import Results
import glob

# In[]
dataset_name = 'Spectralis'  #Spectralis # Cirrus #Topcon

path = "RetouchData/" + dataset_name + "/retouch_data"   #Replace the main path of dataset

print("Dataset: {}".format(dataset_name))

data_path = []
for path in glob.glob(path + '/*'):
    data_path.append(path)    
print("Number of cases : ", len(data_path))

data_reader = DataReader()
# unet = Unet(4, (256,256,1))   # modified: comment out 
loss_funcs = Losses()
my_results = Results()


train_falg = 0
do_continue = False
last_epoch = 20

SEED = 100
NUM_EPOCHS = 10
BATCH_SIZE = 4#*nb_GPUs
BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE

# In[]: Main Loop
kf = KFold(n_splits = 3, shuffle=False)
i = 3
overall_results = []


def decay_schedule(epoch, lr):
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.8
    return lr


for train_path, val_path in kf.split(data_path): 
    tf.random.set_seed(12345)
    if i<=3:
    
        print("Starting Fold number {}".format(i))
        
        train_path, val_path = data_reader.get_trainPath_and_valPath(train_path, val_path, data_path) 
        train_data, val_data = data_reader.get_data_for_train(train_path, val_path)
        num_of_train_samples = len(train_data)
        num_of_val_samples = len(val_data)
        for image, mask in val_data.skip(5).take(1):
            print("Image Shape : ", image.shape)
            print("Mask Shape  : ", mask.shape)
            test_image = image
            test_mask = mask

        # print("Starting Fold number {}".format(i))
        train_data = train_data.shuffle(buffer_size=BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
        val_data = val_data.batch(1).prefetch(buffer_size = AUTOTUNE)
        
        # with strategy.scope():
        # model = unet()        # modified: comment out 
        model = RetiFluidNet(num_class=4, input_shape=(256,256,1))()
        # model.summary()

        initial_learning_rate = 2e-4
        decay_steps = 10000
        decay_rate  = 0.98
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_schedule)

        
        if not os.path.exists(dataset_name):
            os.mkdir(dataset_name)
        
        # Creating Callbacks
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(dataset_name+"/model_%s_checkpoint.hdf5"%i,save_best_only=True) 
                                                            
        
        model.compile(optimizer = tf.keras.optimizers.RMSprop(initial_learning_rate), 
                      loss = loss_funcs.training_loss,
                       metrics = [loss_funcs.dice])        
        if train_falg:
            ival = IntervalEvaluation(validation_data=val_data)
            if do_continue == True:
                model = tf.keras.models.load_model(dataset_name+"/model_%s_epoch%s.hdf5"%(i,last_epoch), custom_objects={'training_loss': loss_funcs.training_loss,
                                                                                                                             'dice_loss': loss_funcs.training_loss,
                                                                                                                             "dice":loss_funcs.dice})
                print("Pre-trained model loaded.")
            history = model.fit(train_data,
                                epochs=NUM_EPOCHS,
                                callbacks=[ival, lr_scheduler])
            model.save(dataset_name+"/model_%s_epoch%s.hdf5"%(i,NUM_EPOCHS))
            with open(dataset_name+"/model_%s_history.npy"%i, 'wb') as f:
                np.save(f, history.history)
        else:
            model = tf.keras.models.load_model(dataset_name+"/model_%s_epoch%s.hdf5"%(i,NUM_EPOCHS), custom_objects={'training_loss': loss_funcs.training_loss,
                                                                                                                             'dice_loss': loss_funcs.training_loss,
                                                                                                                             "dice":loss_funcs.dice})
            try:
                with open(dataset_name+"/model_%s_history.npy"%i, 'rb') as f:
                    History = np.load(f, allow_pickle=True).item()
            except:
                print("No history file is found.") 





        # ======= visualization (AFTER loading/training) =======
        from matplotlib.colors import ListedColormap
        import numpy as np
        import tensorflow as tf

# Colormap
        cmap = ListedColormap(["black", "red", "green", "blue"])

# Select patient and slice
        patient, slice_idx = "TRAIN030", 24
        img_path = f"RetouchData/{dataset_name}/retouch_data/{patient}/{patient}_oct_{slice_idx:03d}.jpg"

# Load image and ground-truth mask
        img, mask = data_reader.load_image(tf.constant(img_path))
        gt_mask = mask.numpy()[..., 0].astype(int)

# ======== START DEBUGGING STEPS ========
        print(f"Image shape: {img.shape}, Mask shape: {gt_mask.shape}, Unique mask labels: {np.unique(gt_mask)}")

# Model prediction (connectivity maps)
        y_pred_full = model.predict(img[None], verbose=0)[0]
        print("Full output shape (should be 256,256,180):", y_pred_full.shape)

        conn_maps = y_pred_full[..., :32]
        print("Connectivity maps shape (should be 256,256,32):", conn_maps.shape)

# Verify each class connectivity map
        for cls in range(4):
            conn_cls = conn_maps[..., cls*8:(cls+1)*8]
            print(f"Class {cls} conn map shape (should be 256,256,8):", conn_cls.shape)
# ======== END DEBUGGING STEPS ========

# Bilateral voting
        prob_maps = []
        for cls in range(4):
            conn_cls = conn_maps[..., cls*8:(cls+1)*8]
            conn_cls_expanded = tf.expand_dims(conn_cls, axis=0)
            prob_map_cls = loss_funcs.bv_test_new(conn_cls_expanded)
            prob_maps.append(prob_map_cls.numpy())

# Stack and predict
        prob_stack = np.stack(prob_maps, axis=-1)
        print("Prob stack shape (should be 256,256,4):", prob_stack.shape)

        pred_mask = np.argmax(prob_stack, axis=-1)
        print("Predicted mask shape:", pred_mask.shape)
        print("Unique labels in pred mask:", np.unique(pred_mask))

# Visualize
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img[..., 0], cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("OCT Image")

        axs[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=3)
        axs[1].axis('off')
        axs[1].set_title("Ground Truth Mask")

        axs[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)
        axs[2].axis('off')
        axs[2].set_title("Predicted Mask")

        plt.tight_layout()
        plt.show()
# =========================================================================










        # plot learning curves
        # if train_falg:
        #     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        #     fig.suptitle('Learning Curves')
            
        #     axs[0].set_title('Model Loss')
        #     axs[0].plot(history.history['loss'], label='train')
        #     axs[0].plot(history.history['val_loss'], label='val')
        #     axs[0].legend()
        #     axs[0].set(xlabel='Epoch', ylabel='Overall-Loss')
            
        #     axs[1].set_title('Model Dice Performance')
        #     axs[1].plot(history.history['main_output_dice_coeff'], label='train')
        #     axs[1].plot(history.history['val_main_output_dice_coeff'], label='val')
        #     axs[1].legend() 
        #     axs[1].set(xlabel='Epoch', ylabel='Main output dice_coeff')
            
        #     plt.show()
        #     fig.savefig(dataset_name+"\model_%s_history.png"%i, dpi=300)
        
        
        # val_data = val_data.take(900)
        
        predictions = []
        for image, mask in tqdm(val_data):  
            temp = model.predict(image)[:, :, :, 0:32]
            predictions.append(temp)
        #print("predictions shape : " , predictions.shape)
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
