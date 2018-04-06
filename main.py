import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import getdata as ld
import models.tracknet as tracknet

parser = opts.myargparser()

def main():
    global opt, best_err1
    opt = parser.parse_args()
    best_err1 = 1000000
    print(opt)
    model = tracknet.Net(opt)
    if opt.cuda:
        model = model.cuda()

    model, criterion, optimizer = init.setup(model, opt)
    print(model)

    trainer = train.Trainer(model, criterion, optimizer, opt)
    # validator = train.Validator(model, criterion, opt)
    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_err1 = init.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    dataloader = ld.SynthLoader(opt)
    train_loader = dataloader.train_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])
        trainer.train(train_loader, epoch, opt)

        # err = validator.validate(val_loader, epoch, opt)
        # best_err1 = min(err, best_err1)
        # print('Best error: [{0:.3f}]\t'.format(best_err1))
        if epoch % 3 == 0 and epoch > 0 and opt.tosave == True:
            init.save_checkpoint(opt, model, optimizer, best_err1, epoch)

if __name__ == '__main__':
    main()
