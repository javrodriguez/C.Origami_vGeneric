import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import os

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

  # Add small_test flag
  parser.add_argument('--small_test', dest='small_test', action='store_true',
                        help='Use only a small subset of chromosomes for testing')

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
        strategy = "ddp" if not args.dataloader_ddp_disabled else "auto"
        devices = args.trainer_num_gpu
    else:
        accelerator = "cpu"
        strategy = "auto"
        devices = 1

    pl_trainer = pl.Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=1,
        logger=all_loggers,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=args.trainer_max_epochs,
        limit_train_batches=args.trainer_limit_train_batches,
        limit_val_batches=args.trainer_limit_val_batches
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
        return self.model(x)

    def proc_batch(self, batch):
        if self.args.use_sequence:
            seq, features, mat, start, end, chr_name, chr_idx = batch
            features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
            inputs = torch.cat([seq, features], dim = 2)
        else:
            features, mat, start, end, chr_name, chr_idx = batch
            features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
            inputs = features
        mat = mat.float()
        return inputs, mat
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        return loss  # Return just the loss value

    # Collect epoch statistics
    def on_train_epoch_end(self):
        # Get the training loss from the logged metrics
        train_loss = self.trainer.callback_metrics.get('train_step_loss', torch.tensor(0.0))
        metrics = {'train_loss': train_loss}
        self.log_dict(metrics, prog_bar=True)

    def on_validation_epoch_end(self):
        # Get the validation loss from the logged metrics
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        metrics = {'val_loss': val_loss}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # First restart epoch
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'CosineAnnealingWarmRestarts',
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

        # If small_test is enabled, use only chr20 for all modes
        chromosomes = ['chr20'] if args.small_test else None

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
