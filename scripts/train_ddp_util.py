import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util
from .fp16_util import MixedPrecisionTrainer


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        seed=9999,
        device='cuda:0',
        rank=0,
        num_workers=4,
        world_size=1
    ):
        self.model = model
        self.diffusion = diffusion
        
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.seed = seed
        self.device = device
        self.rank = rank
        self.training_seed = self.seed + self.rank
        self.num_workers = num_workers
        self.world_size = world_size
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.w = None
        self.train_sampler = th.utils.data.distributed.DistributedSampler(data,
                                                                    num_replicas=self.world_size,
                                                                    rank=rank)
        self.data = th.utils.data.DataLoader(data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            sampler=self.train_sampler,
                                            drop_last = True)
        
        self.iterdatal = iter(self.data)
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        print(f'Rank: {self.rank} - Device: {self.device} {dist.get_rank()}')
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.device]
            )
        else:
            if dist.get_world_size() > 1:
                print(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                print(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        th.manual_seed(self.training_seed)
        th.cuda.manual_seed(self.training_seed)
        th.cuda.manual_seed_all(self.training_seed)
    
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            try:
                batch, cond = next(self.data)
            except:
                self.train_sampler.set_epoch(self.step)
                self.iterdatal = iter(self.data)
                batch, cond = next(self.iterdatal)
            self.run_step(batch, cond)
            # if self.step % self.log_interval == 0:
            #     if self.rank == 0:
            #         logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        if self.rank == 0:
            self.log_step()

    def forward_backward(self, batch, cond):
        # if self.rank == 0:
        #     if self.w is not None:
        #         diff = self.model.out[-1].weight - self.w
        #         print(f'Step {self.step} Before {self.model.out[-1].weight * 10000}')
        #         print(f'Step {self.step} Diff: {diff * 10000}')
        self.mp_trainer.zero_grad()   
        for i in range(0, batch.shape[0], self.microbatch):
            
            micro = batch[i : i + self.microbatch].to(self.device, non_blocking=True)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.device, non_blocking=True)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # print(f'Step {self.step} Rank {self.rank} Data {micro.sum()}')
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

    

            loss = losses["loss"].mean()
            # if self.rank == 0:
            #     log_loss_dict(
            #         self.diffusion, t, {k: v * weights for k, v in losses.items()}
            #     )
            self.mp_trainer.backward(loss)
            # if self.rank == 0:
            # self.w_grad = self.model.out[-1].weight.grad
            # self.w = self.model.out[-1].weight
            # print(f'Step {self.step} After Rank{self.rank}: {loss * 10000}, Loss: {self.w_grad * 10000}')

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        print("step", self.step + self.resume_step)
        print("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if self.rank == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"ckpt/model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ckpt/ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join("logs", filename), "wb") as f:
                    th.save(state_dict, f)
        if self.rank == 0:
            save_checkpoint(0, self.mp_trainer.master_params)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)
    
            with bf.BlobFile(
                bf.join("logs", f"ckpt/opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

def update_ema(target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
