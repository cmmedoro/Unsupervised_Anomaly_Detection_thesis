import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training setting
    parser.add_argument("--train_window", type = int, default = 72,
                        help = "Define a training window size")
    parser.add_argument("--batch_size", type = int, default = 128,
                        help = "Number of windows per iteration")
    parser.add_argument("--hidden_size", type = float, default = 1/8, help = "Size of the hidden layer of the model")
    """
    Model type:
    - usad ---> usad architecture, with encoder and decoders
    - linear_ae ---> simple linear autoencoder (usad with just one decoder)
    - conv_ae ---> convolutional autoencoder
    - lstm_ae ---> lstm autoencoder
    - usad_conv ---> usad with convolution instead of linear layers
    - usad_lstm ---> usad with lstm layers instead of linear layers
    """
    parser.add_argument("--model_type", type = str, default = "usad", help = "Choose a model among usad, conv, lstm")

    parser.add_argument("--epochs", type = int, default = 100)

    parser.add_argument("--do_resid", action = "store_true", help = "Whether to train and test on residuals")
    
    parser.add_argument("--do_train", action = "store_true", help = "Whether to perform training or not")
    parser.add_argument("--do_test", action = "store_true", help = "Whether to perform testing or not") #Overlapping windows
    parser.add_argument("--do_reconstruction", action = "store_true", help = "Whether to perform reconstruction or not") #Non-overlapping windows

    parser.add_argument("--save_checkpoint_dir", type = str, default = None, help = "Indicate folder to store model checkpoint")
    parser.add_argument("--checkpoint_dir", type = str, default = None, help = "Indicate folder to retrieve model checkpoint")
    parser.add_argument("--res_dir", type = str, default = None, help = "Directory to save model outputs")

    parser.add_argument("--threshold", type = int, default = 4, help = "Choose a threshold for anomaly detection")
    parser.add_argument("--percentile", type = float, default = 0.95, help = "Choose a percentile for anomaly detection")
    parser.add_argument("--weights_overall", type = float, default = 0.5, help = "Choose a weight for anomaly detection")

    parser.add_argument("--do_multivariate", action = "store_true", help = "Whether to perform multivariate trials or not")

    args = parser.parse_args()
    return args
