class Config():
    def __init__(self):
        self.dataset = "IXI"
        self.model = "UNET"
        self.mode = "NONE"
        self.loss = "L2"
        self.result = "/scratch/podishet/unet_base_l2/results"
        self.trainlogs = "traininglogs"
        self.validlogs = "validlogs"
        self.images = "tensor_vis"
        self.data = "/scratch/podishet/data/info-data.csv"
        self.original_path = "original_subject.nii.gz"
        self.down_path = "down_subject.nii.gz"
        self.groundtruth = "/scratch/podishet/data/original_normalized"
        self.downsampled = "/scratch/podishet/data/downsampled_interpolated"
        self.inputsize = 10
        self.samples = 25
        self.patchsize = (32, 32, 32)
        self.patchsize_original = (32, 32, 32)
        self.maxqueue = 50
        self.patch_check = "patch.jpg"
        self.learningrate = 0.0001
        self.epochs = 1000
        self.checkpointpath = "/scratch/podishet/unet_base_l2/ckp_unet_L2_base.pt"
        self.bestmodelckp = "/scratch/podishet/unet_base_l2/best_unet_L2_base.pt"
        self.testimage = "test.nii.gz"
        self.test_superres_path = "superres.nii.gz"
        self.test_gt_path = "gt.nii.gz"
        self.test_inp_path = "inp.nii.gz"
        self.training_batch_size = 32
        self.validation_batch_size = 32
        self.test_batch_size = 32
        self.training_split_ratio = 0.85
        self.validation_split_ratio = 0.15
        self.num_features = 64
        self.kernel_size_1 =3
        self.stride_1 =1
        self.kernel_size_2 = 2
        self.stride_2 = 2
        self.kernel_size_3 = 1
        self.scale_factor = 2
        self.num_dimensions = 3