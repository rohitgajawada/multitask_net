from torch.autograd import Variable
from utils import AverageMeter
from copy import deepcopy
import time
import models.__init__ as init
import utils

class Trainer():
    def __init__(self, model, criterion, optimizer, opt):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()

    def train(self, trainloader, epoch, opt):
        self.data_time.reset()
        self.batch_time.reset()
        self.model.train()
        self.losses.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):
            self.optimizer.zero_grad()
            if opt.cuda:
                imgs, annos, names = data
                imgs = imgs.cuda(async=True)
                annos = annos.cuda(async=True)

            imgs, annos = Variable(imgs), Variable(annos)

            self.data_time.update(time.time() - end)

            outputs = self.model(imgs, annos)
            loss = self.criterion(outputs, annos)
            loss.backward()
            self.optimizer.step()

            inputs_size = imgs.size(0)
            self.losses.update(loss.data[0], inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses))

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses))
