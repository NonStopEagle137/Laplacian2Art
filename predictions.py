## Prediction Pipeline for testing GAN Results ##
import numpy as np
import cv2
from generator import *
import tensorflow as tf
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


generator_model = Generator()

class predict:
    def __init__(self,generator_model, in_path,model_path,upscale_model_path,in_image = None,out_dir = None):
        self.model_path = model_path
        self.in_path = in_path
        self.in_image = in_image
        self.out_dir = out_dir
        self.upscale_model_path = upscale_model_path
        self.height = 512
        self.width = 512
        self.generator = generator_model
    def load_model_from_path(self):
        self.generator.load_weights(self.model_path)
        
    def load_image(self):
        if not in_path == None:
            image = cv2.imread(self.in_path)
        else:
            image = self.in_image
        plt.figure()
        plt.imshow(image)
        image = self.resize(image).numpy()
        
        image1 = cv2.Laplacian(image.copy(), cv2.CV_8U, ksize = 5)
        
        image2 = cv2.GaussianBlur(image.copy(), (5,5), 0)
        #image1[np.where(image1 < 15)] = image[np.where(image1 < 15)]
        
        image = image1.copy()
        
        
        
        return image
    def normalize(self, input_image):
        
        input_image = tf.cast(input_image, tf.float32)
        input_image = (input_image / 127.5) - 1
        return input_image
    def resize(self, input_image):
        print(input_image.shape)
        input_image = tf.image.resize(tf.convert_to_tensor(input_image), [self.height, self.width],
                                        method=tf.image.ResizeMethod.BICUBIC)
        
        return input_image
    def denormalize(self, input_image):
        input_image = tf.cast(input_image, tf.float32)
        
        input_image = ((1+ input_image) * 127.5)
        input_image = tf.cast(input_image, tf.uint8)
        #input_image = cv2.cvtColor(input_image.numpy(), cv2.COLOR_BGR2RGB)
        return input_image.numpy()
    def upscale(self, image):
        super_model = cv2.dnn_superres.DnnSuperResImpl_create()
        modelName = 'lapsrn'
        modelScale = 8
        modelPath = self.upscale_model_path
        super_model.readModel(modelPath)
        super_model.setModel(modelName, modelScale)
        image = super_model.upsample(image)
        return image
    def predict_image(self):
        self.load_model_from_path()
        model = self.generator
        #model.summary()
        pre_image = self.load_image()
        
        pre_image = self.normalize(pre_image)
        
        plt.figure()
        plt.imshow(pre_image)
        pred_independent_ = model(np.expand_dims(pre_image, 0), training = True)
        pred_independent = self.denormalize(pred_independent_[0])
        pred_independent = self.upscale(pred_independent)
        return pred_independent
model_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\saved_generator\generator_model.h5'
in_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\Testing\sample3.jpg'
in_image = None

upscale_model_path = r'C:\Users\Athrva Pandhare\Downloads\LapSRN_x8.pb'
  
pred = predict(generator_model, in_path,model_path, upscale_model_path, in_image)
pred__ = pred.predict_image()


pred__ = cv2.cvtColor(pred__, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(pred__)
#cv2.imshow("image", pred__)
pred__ = cv2.cvtColor(pred__, cv2.COLOR_BGR2RGB)
cv2.imwrite(r"C:\Users\Athrva Pandhare\Desktop\New folder (4)\Testing\gen.jpg", pred__)
#cv2.waitKey(0)
#cv2.destroyAllWindows()