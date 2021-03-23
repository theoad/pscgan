import training_methods.gan as gan
import training_methods.mse as mse

factory = {
    'gan': gan.GAN,
    'mse': mse.MSE
}