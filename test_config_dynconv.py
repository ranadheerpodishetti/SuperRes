import dynamic_unet
class Config():
    def __init__(self):
        self.dataset = "IXI"
        self.result = "result"
        self.data = "/scratch/podishet/data/info-test_new.csv"
        self.num_workers = 10
        self.model = dynamic_unet
        self.groundtruth = '/scratch/podishet/data/test_set/Test_Set_gt'
        self.downsampled = '/scratch/podishet/data/test_set/Test_set_inp'
        self.inputsize = 10
        self.patchsize = (96, 96, 48)
        self.patch_overlap = (2, 2, 2)
        self.path = '/scratch/podishet/unet_dynconv_novel_new/output'
        self.gt = '/scratch/podishet/unet_dynconv_novel_new/output'
        self.inp = '/scratch/podishet/unet_dynconv_novel_new/output'
        self.checkpointpath = "/scratch/podishet/wavenet/best_unet_dynconv_novel.pt"
        self.bestmodelckp = '/scratch/podishet/wavenet/best_unet_dynconv_novel.pt'
        self.test_batch_size = 32
        self.num_features = 64
        self.kernel_size_1 = 3
        self.stride_1 = 1
        self.kernel_size_2 = 2
        self.stride_2 = 2
        self.kernel_size_3 = 1
        self.scale_factor = 2
        self.num_dimensions = 3
