import datetime
from transformers.training_args import *


@cached_property
@torch_required
def _setup_devices(self) -> "torch.device":
    logger.info("PyTorch: setting up devices")
    if self.no_cuda:
        device = torch.device("cpu")
        self._n_gpu = 0
    elif is_torch_tpu_available():
        device = xm.xla_device()
        self._n_gpu = 0
    elif is_sagemaker_mp_enabled():
        local_rank = smp.local_rank()
        device = torch.device("cuda", local_rank)
        self._n_gpu = 1
    elif is_sagemaker_dp_enabled():
        sm_dist.init_process_group()
        self.local_rank = sm_dist.get_local_rank()
        device = torch.device("cuda", self.local_rank)
        self._n_gpu = 1
    elif self.deepspeed:
        # deepspeed performs its own DDP internally, and requires the program to be started with:
        # deepspeed  ./program.py
        # rather than:
        # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
        from .deepspeed import is_deepspeed_available

        if not is_deepspeed_available():
            raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
        import deepspeed

        deepspeed.init_distributed()

        # workaround for setups like notebooks where the launcher can't be used,
        # but deepspeed requires a dist env.
        # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

        device = torch.device("cuda", self.local_rank)
        self._n_gpu = 1
    elif self.local_rank == -1:
        # if n_gpu is > 1 we'll use nn.DataParallel.
        # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
        # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
        # trigger an error that a device index is missing. Index 0 takes into account the
        # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
        # will use the first GPU in that env, i.e. GPU#1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
        # the default value.
        self._n_gpu = torch.cuda.device_count()
    else:
        # Here, we'll use torch.distributed.
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        '''THIS IS CHANGED'''
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(0,5400)
        )
        device = torch.device("cuda", self.local_rank)
        self._n_gpu = 1

    if device.type == "cuda":
        torch.cuda.set_device(device)

    return device