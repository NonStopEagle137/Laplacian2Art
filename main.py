import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import datetime
import tensorflow as tf
from loss import *
from generator import *
from discriminator import *
from preprocessing import *
from validity_check import *
from tqdm import tqdm
    
def main(train_model):	
    @tf.function
    def train_step_scratch(input_image, target, epoch, checkpoint):
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

      with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
      return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss
      
    @tf.function
    def train_step(input_image, target, epoch, checkpoint):
      
      with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))

      with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
      return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss
        
    def fit(train_ds, epochs, test_ds):
      
      
      try:
        print("Loaded Checkpoint Successfully...")
        checkpoint.restore(tf.train.latest_checkpoint(r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\Checkpoints'))
        #tf.train.latest_checkpoint(checkpoint_dir)
      except:
        import sys
        print("No Checkpoint found...Starting from scratch") 
        sys.exit()
      for epoch in range(epochs):
        losses_ = np.zeros(4)
        img_count = 0
        start = time.time()
        display.clear_output(wait=True)
        
        for example_input, example_target in test_ds:
          img_count += 1
          if img_count % 50 == 0:
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in tqdm(enumerate(train_ds)):
          losses_ += train_step(np.expand_dims(input_image,0), np.expand_dims(target,0), epoch, checkpoint)
        print(f'Total Loss : {losses_[0]} \n Discriminator Loss : {losses_[1]} \n Generator Loss : {losses_[2]} \n MAE : {losses_[3]}')
        print("=========================")
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                            time.time()-start))
      checkpoint.save(file_prefix=checkpoint_prefix)
      
      
    """Some Variables"""
    file_names = []
    train_dataset_processed = []
    dataset_processed = []
    dataset_transform = []
    test_dataset_processed = []
    train_dataset_transform = []
    test_dataset_transform = []
    
    """Parameters """
    
    EPOCHS = 25
    PATH_TRANSFORM = r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\laplacian'
    PATH_PROCESSED = r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\real'
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    OUTPUT_CHANNELS = 3
    LAMBDA = 500
    
    """Optimizers"""
    
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    # discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    """Processing the dataset"""
    for name in glob.glob(PATH_PROCESSED + '\*.jpg'):
        dataset_processed.append(name)

    for name in glob.glob(PATH_TRANSFORM + '\*.jpg'):
        dataset_transform.append(name)
    # print(int(dataset_processed[7].split('real')[-1][1:-4]))
    # print(dataset_processed)
    
    dataset_processed = sorted(dataset_processed, key = sort_by_number)
    dataset_transform = sorted(dataset_transform, key = sort_by_number)
    

    train_dataset_processed = dataset_processed[:350]
    train_dataset_transform = dataset_transform[:350]
    test_dataset_processed = dataset_processed[350:]
    test_dataset_transform = dataset_transform[350:]


    """Samples should be pix->pix"""
    validity_check_data(dataset_processed, dataset_transform)
    validity_check_data(train_dataset_processed, train_dataset_transform)
    validity_check_data(test_dataset_processed, test_dataset_transform)
    validity_check_data(train_dataset_processed, train_dataset_transform)
    validity_check_data(test_dataset_processed, test_dataset_transform)


    """Loading the dataset into memory"""
    print("[INFO]Loading training and testing data...")
    train_dataset = grab_train_img_from_names(train_dataset_processed, train_dataset_transform)
    test_dataset = grab_train_img_from_names(test_dataset_processed, test_dataset_transform)

    """Base loss object"""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator = Generator()
    print("Generator")
    print("====================")
    generator.summary()
    # discriminator = Discriminator()
    inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar], axis = 1)
    discriminator_loaded = tf.keras.applications.DenseNet121(
                        include_top=False,
                        weights="imagenet",
                        input_tensor=None,
                        input_shape=(1024,512,3),
                        pooling=None,
                        classes=1000,
                        )
    out = discriminator_loaded(x)
    discriminator = tf.keras.Model(inputs = [inp, tar], outputs = out)
    
    print("Discriminator")
    print("====================")
    discriminator.summary()


    checkpoint_dir = r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\Checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                 discriminator_optimizer=discriminator_optimizer,
    #                                 generator=generator,
    #                                 discriminator=discriminator)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    log_dir= r"C:\Users\Athrva Pandhare\Desktop\New folder (4)\logs"

    summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if train_model == True:
        fit(train_dataset, EPOCHS, test_dataset)
    else:
        checkpoint.restore(tf.train.latest_checkpoint(r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\Checkpoints'))
        generator.save(r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\saved_generator\generator_model.h5')
    


if __name__ == "__main__":
  main(train_model = True)
