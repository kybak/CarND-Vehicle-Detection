from library import *
import pickle
from scipy.ndimage.measurements import label
from train_classifier import train_classifier
from moviepy.editor import VideoFileClip



################################### HYPERPARAMETERS ###################################

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
ystart = 400 # Min and max in y to search find_cars()
ystop = 656
scale = 1.5

################################### TRAIN MODEL ###################################


train_classifier(color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)


################################### MULTI-DETECTION SEARCH ###################################

img = mpimg.imread('test_images/test6.jpg')

with open('model.p', 'rb') as f:
    model = pickle.load(f)

svc = model['svc']
X_scaler = model['X_scaler']


def process_image(img):
    image_copy = np.copy(img)
    out_img, detected = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size,
                                  hist_bins)


    heatmap = np.zeros(out_img.shape)
    heatmap = add_heat(heatmap, detected)
    heatmap = threshold(heatmap, 0)
    labels = label(heatmap)

    # Draws bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image_copy), labels)


    return draw_img


################################### PROCESS VIDEO ###################################

test_output = 'output.mp4'
clip = VideoFileClip("project_video.mp4")
test_clip = clip.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)
































