from collections import OrderedDict
import os
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.models import wide_resnet50_2
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from sklearn.metrics import roc_auc_score
from .networks import NetG, weights_init
from .visualizer import Visualizer
from .loss import l2_loss, loss_ssim, MSGMSLoss
from .evaluate import evaluate
import math
from thop import profile

class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader, mask_size):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.cutout_sizes = mask_size

    def mean_smoothing(self, amaps, kernel_size=21):
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
        mean_kernel = mean_kernel.to(amaps.device)
        return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            # self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            self.fixed_input.resize_(input[0].size()).copy_(input[0])
            self.mask.resize_(input[2].size()).copy_(input[2])

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_g', self.err_g.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.fixed_input.data
        fakes = self.netg(self.fixed_input, random.choice(self.cutout_sizes))[0].data
        fixed = self.netg(self.fixed_input, random.choice(self.cutout_sizes))[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            self.set_input(data)
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """
        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        #接下面可视化
        # writer = SummaryWriter('./logs')

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()

            if res[self.opt.metric] > best_auc:

                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)

        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.map_scores = []
            self.gt_mask = []

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)

                losses_an_score = 0
                no_mask_score = 0
                for cutout_size in self.cutout_sizes:
                    self.fake, self.pre_reconst = self.netg(self.input, cutout_size, 1)
                    losses_an_score += torch.mean(torch.abs(self.netf(self.input)-self.netf(self.fake)), dim=1).unsqueeze(1)
                losses_an_score = F.interpolate(losses_an_score, size=(128, 128), mode="bilinear", align_corners=False)
                error = self.mean_smoothing(losses_an_score)
                for id_e in range(error.shape[0]):
                    self.map_scores.append(error[id_e].cpu().detach().numpy())
                    self.gt_mask.append(self.mask[id_e].cpu().detach().numpy())
                
                if self.opt.save_test_images:
                    if self.epoch == 149:
                        dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                        if not os.path.isdir(dst):
                            os.makedirs(dst)
                        real, fake, _ = self.get_current_images()
                        vutils.save_image(real[0], '%s/real_%03d.jpg' % (dst, i + 1), normalize=True)
                        vutils.save_image(fake[0], '%s/fake_%03d.jpg' % (dst, i + 1), normalize=True)

                #下面是没有聚集之后的结果
                # pool_map = 0
                # max_scale = int(math.log2(np.max(self.cutout_sizes)))
                # for pool_scale in range(max_scale + 1):
                #     if pool_scale > 0:
                #         error = F.avg_pool2d(error, kernel_size=2, stride=2, padding=0)
                #     if pool_scale > max_scale - len(self.cutout_sizes):
                #         pool_map += F.interpolate(error, size=(256, 256), mode="bilinear", align_corners=False)

                error = error.reshape(error.size(0), -1)
                error = torch.max(error, axis=1)[0]

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            #定位
            self.map_scores = np.asarray(self.map_scores)
            self.gt_mask = np.asarray(self.gt_mask)
            self.map_scores = (self.map_scores - self.map_scores.min()) / (self.map_scores.max() - self.map_scores.min())
            per_pixel_rocauc = roc_auc_score(self.gt_mask.flatten(), self.map_scores.flatten())
            
            #可视化异常
            if self.epoch == 149:
                for m in range(self.map_scores.shape[0]):
                    if m % 4 == 0:
                        err_heatmap = sns.heatmap(data=self.map_scores[m][0], cmap='jet', square=True, cbar=False, yticklabels=False, xticklabels=False)
                        figs = err_heatmap.get_figure()
                        figs.savefig('%s/HeatMap_%03d.jpg' % (dst, m / 4 + 1), dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        vutils.save_image(torch.tensor(self.gt_mask)[m], '%s/fixed_lable_%03d.jpg' % (dst, m / 4 + 1), normalize=True)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc), ('piex roc', per_pixel_rocauc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader, mask_size):
        super(Ganomaly, self).__init__(opt, dataloader, mask_size)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netf = wide_resnet50_2(pretrained=True, progress=True)
        self.netf = nn.Sequential(self.netf.conv1, self.netf.bn1, self.netf.relu, self.netf.maxpool, self.netf.layer1)

        self.netf.to(self.device)
        self.netf.eval()

        self.netg.apply(weights_init)
        # self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])

        # 检查参数
        inpu = torch.randn(1, 3, 128, 128).cuda(0)
        flops, params = profile(self.netg, (inpu, 2))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

        self.l_adv = l2_loss
        self.l_con = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.mask = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.pre_reconst = self.netg(self.input, random.choice(self.cutout_sizes))

    ##
    def total_loss(self, img1, img2):
        err_g_adv = self.l_adv(self.netf(img1), self.netf(img2))
        err_g_con = self.l_con(img1, img2)
        err_g = 20 * err_g_con + 10 * err_g_adv
        return err_g
    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g = self.total_loss(self.input, self.fake) + self.total_loss(self.input, self.pre_reconst)
        self.err_g.backward()
    ##

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
