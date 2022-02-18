import dynamic_unet
class Config():
    def __init__(self):
        self.dataset = "IXI"
        self.result = "result"
        self.data = "E:/master/sem2/superres/execution/info-test_new.csv"
        self.model = dynamic_unet
        self.groundtruth = 'D:/test_gt'
        self.downsampled = 'D:/test_inp'
        self.inputsize = 10
        self.patchsize = (32, 32, 32)
        self.patch_overlap = (2, 2, 2)
        self.path = 'E:/master/sem2/superres/execution/results/super_l2.nii.gz'
        self.gt = 'E:/master/sem2/superres/execution/results/gt.nii.gz'
        self.inp = 'E:/master/sem2/superres/execution/results/inp.nii.gz'
        self.checkpointpath = "ckp_final_unet_l1.pt"
        self.bestmodelckp = 'D:/checkpoints/unet_l1_deepsup/ckp1/ckp_unet_l1_deepsup.pt'
        self.test_batch_size = 32
        self.num_features = 64
        self.kernel_size_1 = 3
        self.stride_1 = 1
        self.kernel_size_2 = 2
        self.stride_2 = 2
        self.kernel_size_3 = 1
        self.scale_factor = 2
        self.num_dimensions = 3