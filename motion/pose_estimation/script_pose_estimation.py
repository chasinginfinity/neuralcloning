from demo_image import process
from model import get_testing_model
import os
from config_reader import config_reader
import cv2

# get images from 'bruno frames'
all_images = os.listdir('bruno frames')

keras_weights_file = 'model/keras/model.h5'

model = get_testing_model()
model.load_weights(keras_weights_file)

# load config
params, model_params = config_reader()

for image in all_images[:5]:
    input_image = "bruno frames/" + image
    output = "bruno poses/" + image

    print(input_image)

    # tic = time.time()
    # print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images    

    # generate image with body parts
    canvas = process(input_image, params, model_params, model)

    # toc = time.time()
    # print ('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

cv2.destroyAllWindows()