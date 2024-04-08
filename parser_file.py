import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training setting
    parser.add_argument("--train_window", type = int, default = 72,
                        help = "Define a training window size")
    parser.add_argument("--batch_size", type = int, default = 128,
                        help = "Number of windows per iteration")
    parser.add_argument("--hidden_size", type = float, default = 1/8, help = "Size of the hidden layer of the model")
    parser.add_argument("--model_type", type = bool, default = "usad", help = "Choose a model among usad, conv, lstm")

    parser.add_argument("--epochs", type = int, default = 100)

    parser.add_argument("--do_train", action = "store_true", help = "Whether to perform training or not")
    parser.add_argument("--do_test", action = "store_true", help = "Whether to perform testing or not")

    parser.add_argument("--save_checkpoint_dir", type = str, default = None, help = "Indicate folder to store model checkpoint")
    parser.add_argument("--checkpoint_dir", type = str, default = None, help = "Indicate folder to retrieve model checkpoint")
    
    args = parser.parse_args()
    return args
