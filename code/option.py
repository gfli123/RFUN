import argparse

parser = argparse.ArgumentParser(description='Pytorch 1.8.1: Model Example')
parser.add_argument('--batch_size', type=int, default=1, help="Training batch size. default=8")
parser.add_argument('--crop_size', type=int, default=256, help="Training image cropping size")
parser.add_argument('--clip', type=int, default=20, help="Training image cropping size")
parser.add_argument('--gamma', type=int, default=0.1, help="Learning rate decay")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate. default=0.0001")
parser.add_argument('--image_out', type=str, default='./results_out', help="Test data")
parser.add_argument('--num_epochs', type=int, default=90, help="Number of epochs to train")
parser.add_argument('--num_frames', type=int, default=7, help="Several image in a group")
parser.add_argument('--save_model_path', type=str, default='./checkpoints', help="Location to save checkpoint models")
parser.add_argument('--scale', type=int, default=4, help="Super resolution upscale factor")
parser.add_argument('--seed', type=int, default=6, help="Random seed")
parser.add_argument('--start_epoch', type=int, default=0, help="Start place, default=0")
parser.add_argument('--step_size', type=int, default=70, help='Learning rate is decayed by a factor of 10000 epochs')
parser.add_argument('--test_data', type=str, default='./data_test/filelist_test.txt', help="Test data")
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--train_data', type=str, default='./data81/sep_trainlist.txt', help="Training data")
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay")

opt = parser.parse_args()

