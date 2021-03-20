import numpy as np
import pandas as pd
import yaml
import os
import datetime
import cv2
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from src.data.preprocess import remove_text
from shap import GradientExplainer
from shap.plots import colors

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

def plot(shap_values, pixel_values, img_name, truth, labels, aspect=0.2, hspace='auto', dir_path=None):
    """ Plots SHAP values for image inputs.
    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.
    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.
    labels : list
        List of names for each of the model outputs that are being explained. This list should be the same length
        as the shap_values list.
    """

    multi_output = True
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    sv_len = len(shap_values)
    
    # make sure labels
    assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    if multi_output:
        assert labels.shape[1] == sv_len, "Labels must have a column for each output in shap_values!"
    else:
        assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    # plot our explanations
    x = pixel_values
    fig_size = np.array([5 * (sv_len + 1), 5 * (x.shape[0] + 1)])
    
    fig, ax = plt.subplots(nrows=x.shape[0], ncols=sv_len + 1, figsize=fig_size)
    if len(ax.shape) == 1:
        ax = ax.reshape(1, ax.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr = x_curr / 255

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
        else:
            x_curr_gray = x_curr

        ax[row,0].set_title(truth)
        ax[row,0].imshow((x_curr * 255).astype(np.uint8))
        ax[row,0].axis('off')

        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(sv_len)], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(sv_len)], 0).flatten()
        
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(sv_len):
            ax[row,i+1].set_title(labels[row,i])
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            ax[row,i+1].imshow((x_curr_gray * 255).astype(np.uint8), cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[1], sv.shape[0], -1))
            img = ax[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            ax[row,i+1].axis('off')
    
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
        
    cb = fig.colorbar(img, ax=np.ravel(ax).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)

    # Save the image
    if dir_path is not None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file = dir_path + img_name.split('/')[-1] + '_exp_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
        plt.savefig(file)

    plt.show()

def setup_shap():
    '''
    Load relevant information and create a SHAP Explainer
    :return: dict containing important information and objects for explanation experiments
    '''
    
    # Load relevant constants from project config file
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    shap_dict = {}
    shap_dict['IMG_PATH'] = cfg['PATHS']['IMAGES']
    shap_dict['RAW_DATA_PATH'] = cfg['PATHS']['RAW_DATA']
    shap_dict['IMG_DIM'] = cfg['DATA']['IMG_DIM']
    shap_dict['CLASSES'] = cfg['DATA']['CLASSES']

    # Load train and test sets
    shap_dict['TRAIN_SET'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    shap_dict['TEST_SET'] = pd.read_csv(cfg['PATHS']['TEST_SET'])
    
    row_count = len(shap_dict['TRAIN_SET'].index)
    data = []
    for i in range(row_count):
        # Get the corresponding original image (no preprocessing)
        orig_img = cv2.imread(shap_dict['RAW_DATA_PATH'] + shap_dict['TRAIN_SET']['filename'][i])
        new_dim = tuple(shap_dict['IMG_DIM'])
        orig_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_AREA)    # Resize image
        data.append(orig_img)
    data = np.array(data)
    
    # Load trained model's weights
    shap_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    # Define the SHAP explainer
    shap_dict['EXPLAINER'] = GradientExplainer(shap_dict['MODEL'], preprocess_input(data.copy()), batch_size=64)
    return shap_dict

def explain_xray(shap_dict, idx, save_exp=True):
    '''
    Make a prediction and provide a SHAP explanation
    :param shap_dict: dict containing important information and objects for explanation experiments
    :param idx: index of image in test set to explain
    :param save_exp: Boolean indicating whether to save the explanation visualization
    '''

    # Get the corresponding original image (no preprocessing)
    orig_img = cv2.imread(shap_dict['RAW_DATA_PATH'] + shap_dict['TEST_SET']['filename'][idx])
    new_dim = tuple(shap_dict['IMG_DIM'])
    orig_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_AREA)    # Resize image
    to_explain = np.expand_dims(orig_img, axis=0)

    # Make a prediction for this image and retrieve a SHAP explanation for the prediction
    start_time = datetime.datetime.now()
    shap_values, indexes = shap_dict['EXPLAINER'].shap_values(preprocess_input(to_explain), ranked_outputs=1)
    print("Explanation time = " + str((datetime.datetime.now() - start_time).total_seconds()) + " seconds")

    # Get image filenames, ground truth and prediction labels
    img_name = shap_dict['TEST_SET']['filename'][idx]
    truth = shap_dict['TEST_SET']['label_str'][idx]
    labels = np.vectorize(lambda x: shap_dict['CLASSES'][x])(indexes)

    # Visualize the SHAP explanation and optionally save it to disk
    path = None
    if save_exp:
        path = shap_dict['IMG_PATH']
    
    plot(shap_values, to_explain, img_name, truth, labels, dir_path=path)

def main():
    shap_dict = setup_shap()
    row_count = len(shap_dict['TEST_SET'].index)
    for i in range(row_count):                       # Select i'th image in test set
        explain_xray(shap_dict, i, save_exp=False)    # Generate explanation for image