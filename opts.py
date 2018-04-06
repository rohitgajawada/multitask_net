import argparse
optim_choices = ['sgd','adam','adagrad', 'adamax', 'adadelta']

def myargparser():
    parser = argparse.ArgumentParser(description='GazeNet Training')

    #data stuff
    parser.add_argument('--dataset', default='synth_football', type=str, help='chosen dataset')
    parser.add_argument('--data_dir', default='./synth_football/', type=str, help='chosen data directory')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--workers', default=6, type=int, help='number of data loading workers (default: 4)')
    #default stuff
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--testbatchsize', default=128, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--printfreq', default=1, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayschedular', type=str, help='if lr rate scheduler should be used')

    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', default=10, type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', default=1.15, type=int, help='decays by a power of decaylevel')
    parser.add_argument('--criterion', default='l1', help='Criterion')
    parser.add_argument('--optimType', default='adam', choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', default=0.0001, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    #extra model stuff
    parser.add_argument('--model_def', default='tracknet', help='Architectures to be loaded')
    parser.add_argument('--img_size', default=(3, 224, 224), type=int, help='Input size')
    #default
    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda',  default=True, help='if cuda is available')
    parser.add_argument('--manualSeed',  default=123, help='fixed seed for experiments')
    parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', default="")
    parser.add_argument('--tosave', default=False)

    #model stuff
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str,
                        help='path to storing checkpoints (default: none)')


    return parser
