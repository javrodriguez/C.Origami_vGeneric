import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import os
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.profilers import PyTorchProfiler

import corigami.model.corigami_models as corigami_models
from corigami.data import genome_dataset

def main():
    args = init_parser()
    init_training(args)

def init_parser():
  def str_or_none(value):
    if value.lower() == 'none':
      return None
    return value

  parser = argparse.ArgumentParser(description='C.Origami Training Module.')

  # Data and Run Directories
  parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

  # Data directories
  parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
  parser.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                        help='Genome assembly for training data')
  parser.add_argument('--celltype', dest='dataset_celltype', default='imr90',
                        help='Comma-separated list of cell types (e.g., "BALL-MCG001,BALL-MCG003")')

  # Model parameters
  parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                        help='CNN with Transformer')

  # Training Parameters
  parser.add_argument('--patience', dest='trainer_patience', default=80,
                        type=int,
                        help='Epoches before early stopping')
  parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=80,
                        type=int,
                        help='Max epochs')
  parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                        type=int,
                        help='Top n models to save')
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
                        type=int,
                        help='Number of GPUs to use')
  parser.add_argument('--limit_train_batches', dest='trainer_limit_train_batches', default=None,
                        type=int,
                        help='Limit number of training batches per epoch')
  parser.add_argument('--limit_val_batches', dest='trainer_limit_val_batches', default=None,
                        type=int,
                        help='Limit number of validation batches per epoch')

  # Dataloader Parameters
  parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, 
                        type=int,
                        help='Batch size')
  parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
  parser.add_argument('--num-workers', dest='dataloader_num_workers', default=16,
                        type=int,
                        help='Dataloader workers')

  # New argument for global genomic feature normalization
  parser.add_argument('--genomic_features_norm', dest='genomic_features_norm', default=None,
                        choices=[None, 'log'],
                        type=str_or_none,  # Use the custom type converter
                        help='Global normalization method for genomic features')

  # Add argument for feature-specific normalization
  parser.add_argument('--feature_norms', dest='feature_norms', default=None,
                        type=str,
                        help='Comma-separated list of feature:norm pairs (e.g., "ctcf:log,atac:None")')

  # Add test chromosome argument
  parser.add_argument('--test_chromosome', dest='test_chromosome', nargs='?', const='chr20', default=None,
                        help='Use a single chromosome for testing. Without value uses chr20, with value uses specified chromosome (e.g., --test_chromosome chr1)')

  # Add sequence usage control
  parser.add_argument('--no-sequence', dest='use_sequence', action='store_false', default=True,
                        help='Disable DNA sequence as a feature (default: sequence is used)')

  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  return args

def init_training(args):
    # Validate cell types first
    assembly_root = f"{args.dataset_data_root}/{args.dataset_assembly}"
    dataset_ids = [id.strip() for id in args.dataset_celltype.split(',')]
    
    # Validate all cell types exist
    for dataset_id in dataset_ids:
        celltype_root = f"{assembly_root}/{dataset_id}"
        if not os.path.exists(celltype_root):
            raise ValueError(f"Cell type directory not found: {celltype_root}")

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
    all_loggers = csv_logger
    
    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)

    # Determine accelerator and strategy
    if args.trainer_num_gpu > 0:
        accelerator = "gpu"
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False) if not args.dataloader_ddp_disabled else "auto"
        devices = args.trainer_num_gpu
    else:
        accelerator = "cpu"
        strategy = "auto"
        devices = 1

    # Configure profiler with more detailed settings
    profiler = PyTorchProfiler(
        dirpath=f'{args.run_save_path}/profiler',
        filename='profile',
        export_to_chrome=True,
        row_limit=100,
        sort_by_key='cuda_time_total',
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{args.run_save_path}/profiler/tensorboard'),
        record_shapes=True,
        with_flops=True,
        with_modules=True
    )

    pl_trainer = pl.Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=1,
        logger=all_loggers,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=args.trainer_max_epochs,
        limit_train_batches=args.trainer_limit_train_batches,
        limit_val_batches=args.trainer_limit_val_batches,
        profiler=profiler
    )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        with record_function("model_forward"):
            # Encoder profiling
            with record_function("encoder_conv_start"):
                x = self.model.encoder.conv_start(x)
            with record_function("encoder_res_blocks"):
                x = self.model.encoder.res_blocks(x)
            with record_function("encoder_conv_end"):
                x = self.model.encoder.conv_end(x)
            
            # Transformer profiling
            with record_function("transformer_attention"):
                x = self.model.attn(x)
            
            # Decoder profiling
            with record_function("decoder_conv_start"):
                x = self.model.decoder.conv_start(x)
            with record_function("decoder_res_blocks"):
                x = self.model.decoder.res_blocks(x)
            with record_function("decoder_conv_end"):
                x = self.model.decoder.conv_end(x)
            return x

    def proc_batch(self, batch):
        with record_function("proc_batch"):
            if self.args.use_sequence:
                with record_function("sequence_loading"):
                    seq, features, mat, start, end, chr_name, chr_idx = batch
                with record_function("feature_processing"):
                    features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
                    inputs = torch.cat([seq, features], dim = 2)
            else:
                with record_function("feature_loading"):
                    features, mat, start, end, chr_name, chr_idx = batch
                with record_function("feature_processing"):
                    features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
                    inputs = features
            with record_function("matrix_processing"):
                mat = mat.float()
            return inputs, mat
    
    def training_step(self, batch, batch_idx):
        with record_function("training_step"):
            with record_function("data_loading"):
                inputs, mat = self.proc_batch(batch)
            with record_function("model_forward"):
                outputs = self(inputs)
            with record_function("loss_computation"):
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, mat)
            self.log('train_step_loss', loss, batch_size=inputs.shape[0], sync_dist=True, on_step=True, on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        with record_function("validation_step"):
            with record_function("data_loading"):
                inputs, mat = self.proc_batch(batch)
            with record_function("model_forward"):
                outputs = self(inputs)
            with record_function("loss_computation"):
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, mat)
            self.log('val_loss', loss, batch_size=inputs.shape[0], sync_dist=True)
            return loss

    def test_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        self.log('test_loss', loss, batch_size=inputs.shape[0], sync_dist=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        return loss

    # Collect epoch statistics
    def on_train_epoch_end(self):
        # Get the training loss from the logged metrics
        train_loss = self.trainer.callback_metrics.get('train_step_loss', torch.tensor(0.0))
        self.log('train_loss', train_loss, batch_size=self.args.dataloader_batch_size, sync_dist=True)

    def on_validation_epoch_end(self):
        # No need to log val_loss here as it's already logged in validation_step
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,  # Maximum learning rate
                                     weight_decay = 0)

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_epochs = 10
        total_epochs = self.args.trainer_max_epochs
        min_lr = 1e-6
        max_lr = 2e-4

        # Linear warmup from min_lr to max_lr for the first 10 epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=min_lr/max_lr,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # Cosine annealing for the rest
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        # Combine them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'LinearWarmup+CosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):
        assembly_root = f"{args.dataset_data_root}/{args.dataset_assembly}"
        dataset_ids = [id.strip() for id in args.dataset_celltype.split(',')]

        # Validate all cell types and their features first
        required_features = None
        for dataset_id in dataset_ids:
            celltype_root = f"{assembly_root}/{dataset_id}"
            if not os.path.exists(celltype_root):
                raise ValueError(f"Cell type directory not found: {celltype_root}")
            
            # Get features for this cell type
            features = self.get_genomic_features(args, celltype_root)
            
            # Check feature consistency
            if required_features is None:
                required_features = set(features.keys())
            elif set(features.keys()) != required_features:
                raise ValueError(f"Cell type {dataset_id} has different features than {dataset_ids[0]}. "
                               f"Expected: {required_features}, Got: {set(features.keys())}")

        # Set chromosomes based on mode
        if mode == 'train' and args.test_chromosome:
            # For training, use the specified test chromosome if provided
            chromosomes = [args.test_chromosome]
        elif mode == 'val':
            # For validation, always use chr10
            chromosomes = ['chr10']
        else:
            # For test or when no test_chromosome is specified, use all chromosomes
            chromosomes = None

        # Create datasets only after validation
        datasets = [
            genome_dataset.GenomeDataset(
                f"{assembly_root}/{dataset_id}",
                args.dataset_assembly,
                self.get_genomic_features(args, f"{assembly_root}/{dataset_id}"),
                mode=mode,
                include_sequence=args.use_sequence,
                include_genomic_features=True,
                chromosomes=chromosomes
            )
            for dataset_id in dataset_ids
        ]
        dataset = torch.utils.data.ConcatDataset(datasets)

        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_genomic_features(self, args, celltype_root):
        genomic_features_dir = f'{celltype_root}/genomic_features'
        bw_files = [f for f in os.listdir(genomic_features_dir) if f.endswith('.bw')]

        # Parse feature-specific normalizations if provided
        feature_norms = {}
        if args.feature_norms:
            for pair in args.feature_norms.split(','):
                feature, norm = pair.split(':')
                if norm.lower() == 'none':
                    norm = None
                feature_norms[feature] = norm

        genomic_features = {}
        for bw_file in bw_files:
            feature_name = bw_file.replace('.bw', '')
            # Use feature-specific norm if provided, otherwise use global norm
            norm = feature_norms.get(feature_name, args.genomic_features_norm)
            genomic_features[feature_name] = {'file_name': bw_file, 'norm': norm}

        return genomic_features

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if mode == 'train':
            shuffle = True
        else: # validation and test settings
            shuffle = False
        
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled and args.trainer_num_gpu > 0:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader

    def get_model(self, args):
        model_name = args.model_type
        # Use the first cell type for model initialization
        first_celltype = args.dataset_celltype.split(',')[0].strip()
        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{first_celltype}'
        genomic_features_dir = f'{celltype_root}/genomic_features'
        num_genomic_features = len([f for f in os.listdir(genomic_features_dir) if f.endswith('.bw')])
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 256, use_sequence = args.use_sequence)
        return model

if __name__ == '__main__':
    main()
