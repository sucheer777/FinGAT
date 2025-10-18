# """
# Enhanced Training Script with Complete Metric Tracking
# """

# import argparse
# import os
# import yaml
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# # Import modules
# from data.data_loader import FinancialDataset
# from training.lightning_module import FinGATLightningModule, FinGATDataModule
# from evaluation.metrics import FinancialMetrics


# class MetricsTracker:
#     """Track and save training metrics"""
    
#     def __init__(self, save_dir: str = "results"):
#         self.save_dir = save_dir
#         os.makedirs(save_dir, exist_ok=True)
        
#     def extract_metrics(self, trainer):
#         """Extract metrics from trainer logs"""
        
#         # Get logged metrics
#         logged_metrics = trainer.logged_metrics
        
#         # Extract metrics by epoch from logger
#         if hasattr(trainer.logger, 'log_dir'):
#             # For TensorBoard logger
#             metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
#             if os.path.exists(metrics_path):
#                 metrics_df = pd.read_csv(metrics_path)
#                 return metrics_df
        
#         return pd.DataFrame([logged_metrics])
    
#     def plot_metrics(self, metrics_df):
#         """Plot training curves"""
        
#         if metrics_df.empty:
#             print("No metrics to plot")
#             return
            
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle('FinGAT Training Metrics', fontsize=16)
        
#         # Loss curves
#         if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
#             axes[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
#             axes[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
#             axes[0, 0].set_title('Loss Curves')
#             axes[0, 0].set_xlabel('Epoch')
#             axes[0, 0].set_ylabel('Loss')
#             axes[0, 0].legend()
#             axes[0, 0].grid(True)
        
#         # Accuracy curves
#         if 'train_accuracy' in metrics_df.columns and 'val_accuracy' in metrics_df.columns:
#             axes[0, 1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], label='Train Accuracy')
#             axes[0, 1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Val Accuracy')
#             axes[0, 1].set_title('Classification Accuracy')
#             axes[0, 1].set_xlabel('Epoch')
#             axes[0, 1].set_ylabel('Accuracy')
#             axes[0, 1].legend()
#             axes[0, 1].grid(True)
        
#         # Regression metrics
#         if 'train_mae' in metrics_df.columns and 'val_mae' in metrics_df.columns:
#             axes[0, 2].plot(metrics_df['epoch'], metrics_df['train_mae'], label='Train MAE')
#             axes[0, 2].plot(metrics_df['epoch'], metrics_df['val_mae'], label='Val MAE')
#             axes[0, 2].set_title('Mean Absolute Error')
#             axes[0, 2].set_xlabel('Epoch')
#             axes[0, 2].set_ylabel('MAE')
#             axes[0, 2].legend()
#             axes[0, 2].grid(True)
        
#         # R2 Score
#         if 'train_r2' in metrics_df.columns and 'val_r2' in metrics_df.columns:
#             axes[1, 0].plot(metrics_df['epoch'], metrics_df['train_r2'], label='Train R²')
#             axes[1, 0].plot(metrics_df['epoch'], metrics_df['val_r2'], label='Val R²')
#             axes[1, 0].set_title('R² Score')
#             axes[1, 0].set_xlabel('Epoch')
#             axes[1, 0].set_ylabel('R²')
#             axes[1, 0].legend()
#             axes[1, 0].grid(True)
        
#         # MRR (Ranking Quality)
#         if 'val_mrr' in metrics_df.columns:
#             axes[1, 1].plot(metrics_df['epoch'], metrics_df['val_mrr'], label='Val MRR')
#             axes[1, 1].set_title('Mean Reciprocal Rank')
#             axes[1, 1].set_xlabel('Epoch')
#             axes[1, 1].set_ylabel('MRR')
#             axes[1, 1].legend()
#             axes[1, 1].grid(True)
        
#         # Top-K Precision
#         if 'val_precision@10' in metrics_df.columns:
#             axes[1, 2].plot(metrics_df['epoch'], metrics_df['val_precision@10'], label='Precision@10')
#             if 'val_precision@5' in metrics_df.columns:
#                 axes[1, 2].plot(metrics_df['epoch'], metrics_df['val_precision@5'], label='Precision@5')
#             axes[1, 2].set_title('Top-K Precision')
#             axes[1, 2].set_xlabel('Epoch')
#             axes[1, 2].set_ylabel('Precision')
#             axes[1, 2].legend()
#             axes[1, 2].grid(True)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
#         plt.show()


# def main():
#     """Enhanced main training function"""
    
#     parser = argparse.ArgumentParser(description='Train Modern FinGAT 2025')
#     parser.add_argument('--config', type=str, default='config/config.yaml')
#     parser.add_argument('--fast_dev_run', action='store_true')
    
#     args = parser.parse_args()
    
#     print("="*60)
#     print("🚀 MODERN FINGAT 2025 - ENHANCED TRAINING")
#     print("="*60)
    
#     # Set random seeds
#     pl.seed_everything(42, workers=True)
    
#     # Create directories
#     os.makedirs("checkpoints", exist_ok=True)
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)
    
#     # Prepare dataset
#     print("📊 Preparing dataset...")
#     dataset = FinancialDataset(
#         csv_folder_path="data\stocks",
#         max_stocks=550 
#     )
#     data, metadata = dataset.prepare_dataset()
    
#     print(f"✅ Dataset prepared:")
#     print(f"   📈 Number of stocks: {data.num_nodes}")
#     print(f"   🔢 Features per stock: {metadata['num_features']}")  
#     print(f"   🔗 Graph edges: {data.edge_index.size(1)}")
    
#     # Create model configuration
#     config = {
#         'model': {
#             'hidden_dim': 128,
#             'num_heads': 8,
#             'num_layers': 3,
#         },
#         'training': {
#             'learning_rate': 0.001,
#             'weight_decay': 0.01,
#             'max_epochs': 50,
#             'patience': 15
#         }
#     }

#     model = FinGATLightningModule(config, metadata)
#     data_module = FinGATDataModule(config, data, metadata)

#     tb_logger = TensorBoardLogger("logs", name="fingat")
#     csv_logger = CSVLogger("logs", name="fingat")

#     checkpoint_callback = ModelCheckpoint(
#         dirpath="checkpoints",
#         filename='fingat-{epoch:02d}-{val_mrr:.4f}',
#         monitor='val_mrr',
#         mode='max',
#         save_top_k=3,
#         save_last=True
#     )
    
#     early_stop_callback = EarlyStopping(
#         monitor='val_mrr',
#         patience=config['training']['patience'],
#         mode='max',
#         verbose=True
#     )
    
#     # Create trainer
#     trainer = pl.Trainer(
#         max_epochs=config['training']['max_epochs'],
#         accelerator='auto',
#         devices='auto',
#         callbacks=[checkpoint_callback, early_stop_callback],
#         logger=[tb_logger, csv_logger],
#         fast_dev_run=args.fast_dev_run,
#         enable_progress_bar=True,
#         log_every_n_steps=1,
#         check_val_every_n_epoch=1
#     )

#     print("🏋️ Starting training...")
#     trainer.fit(model, data_module)

#     print("🧪 Running final evaluation...")
#     test_results = trainer.test(model, data_module, ckpt_path='best')

#     metrics_tracker = MetricsTracker("results")
#     metrics_df = metrics_tracker.extract_metrics(trainer)
    
#     if not metrics_df.empty:
#         metrics_df.to_csv("results/training_metrics.csv", index=False)
#         print("📊 Metrics saved to results/training_metrics.csv")
        
#         metrics_tracker.plot_metrics(metrics_df)
#         print("📈 Training plots saved to results/training_metrics.png")
    
#     print("\n✅ Training completed successfully!")
#     print(f"🏆 Best model: {checkpoint_callback.best_model_path}")

#     print("\n📋 **FINAL TEST RESULTS:**")
#     print("-"*50)
#     for key, value in trainer.logged_metrics.items():
#         if key.startswith('test_'):
#             print(f"{key:<25}: {value:.4f}")


# if __name__ == '__main__':
#     main()


"""
FIXED Training Script with Proper Data Splits
"""

import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import modules
from data.data_loader import FinancialDataset
from training.lightning_module import FinGATLightningModule, FinGATDataModule


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def extract_metrics(self, trainer):
        """Extract metrics from trainer logs"""
        logged_metrics = trainer.logged_metrics
        
        if hasattr(trainer.logger, 'log_dir'):
            metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                return metrics_df
        
        return pd.DataFrame([logged_metrics])
    
    def plot_metrics(self, metrics_df):
        """Plot training curves"""
        
        if metrics_df.empty:
            print("No metrics to plot")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FinGAT Training Metrics', fontsize=16)
        
        # Loss curves
        if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
            axes[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
            axes[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if 'train_accuracy' in metrics_df.columns and 'val_accuracy' in metrics_df.columns:
            axes[0, 1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], label='Train Accuracy')
            axes[0, 1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Classification Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Regression metrics
        if 'train_mae' in metrics_df.columns and 'val_mae' in metrics_df.columns:
            axes[0, 2].plot(metrics_df['epoch'], metrics_df['train_mae'], label='Train MAE')
            axes[0, 2].plot(metrics_df['epoch'], metrics_df['val_mae'], label='Val MAE')
            axes[0, 2].set_title('Mean Absolute Error')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('MAE')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # R2 Score
        if 'train_r2' in metrics_df.columns and 'val_r2' in metrics_df.columns:
            axes[1, 0].plot(metrics_df['epoch'], metrics_df['train_r2'], label='Train R²')
            axes[1, 0].plot(metrics_df['epoch'], metrics_df['val_r2'], label='Val R²')
            axes[1, 0].set_title('R² Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # MRR (Ranking Quality)
        if 'val_mrr' in metrics_df.columns:
            axes[1, 1].plot(metrics_df['epoch'], metrics_df['val_mrr'], label='Val MRR')
            axes[1, 1].set_title('Mean Reciprocal Rank')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MRR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Top-K Precision
        if 'val_precision@10' in metrics_df.columns:
            axes[1, 2].plot(metrics_df['epoch'], metrics_df['val_precision@10'], label='Precision@10')
            if 'val_precision@5' in metrics_df.columns:
                axes[1, 2].plot(metrics_df['epoch'], metrics_df['val_precision@5'], label='Precision@5')
            axes[1, 2].set_title('Top-K Precision')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Precision')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """✅ FIXED main training function with proper splits"""
    
    parser = argparse.ArgumentParser(description='Train Modern FinGAT 2025')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--fast_dev_run', action='store_true')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 MODERN FINGAT 2025 - FIXED TRAINING")
    print("="*60)
    
    # Set random seeds
    pl.seed_everything(42, workers=True)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # ✅ FIX 1: Prepare dataset
    print("📊 Preparing dataset...")
    dataset = FinancialDataset(
        csv_folder_path="indian_data",  # Fixed path separator
        max_stocks=550 
    )
    data, metadata = dataset.prepare_dataset()
    
    # ✅ FIX 2: CREATE PROPER TRAIN/VAL/TEST SPLITS
    print("\n📊 Creating train/val/test splits...")
    train_data, val_data, test_data = dataset.create_temporal_splits(
        data, 
        metadata,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"\n✅ Dataset prepared:")
    print(f"   📈 Train stocks: {train_data.num_nodes}")
    print(f"   📊 Val stocks: {val_data.num_nodes}")
    print(f"   🧪 Test stocks: {test_data.num_nodes}")
    print(f"   🔢 Features per stock: {metadata['num_features']}")  
    print(f"   🔗 Total graph edges: {data.edge_index.size(1)}")
    print(f"   🔗 Train edges: {train_data.edge_index.size(1)}")
    
    # ✅ FIX 3: Updated model configuration with better hyperparameters
    config = {
    'model': {
        'hidden_dim': 128,  # ✅ Keep small
        'num_heads': 4,     # ✅ Keep small
        'num_layers': 2,    # ✅ Keep shallow
        'dropout': 0.2      # ✅ Keep regularization
    },
    'training': {
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'max_epochs': 50,
        'patience': 30
    }
}


    # Initialize model
    model = FinGATLightningModule(config, metadata)
    
    # ✅ FIX 4: Initialize data module with SPLITS
    data_module = FinGATDataModule(
        config=config,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metadata=metadata
    )

    # Setup loggers
    tb_logger = TensorBoardLogger("logs", name="fingat")
    csv_logger = CSVLogger("logs", name="fingat")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename='fingat-{epoch:02d}-{val_mrr:.4f}',
        monitor='val_mrr',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_mrr',
        patience=config['training']['patience'],
        mode='max',
        verbose=True
    )
    
    # ✅ FIX 5: Create trainer with gradient clipping
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,  # ✅ Prevent exploding gradients
        deterministic=True       # ✅ Reproducibility
    )

    # Train model
    print("\n🏋️ Starting training...")
    trainer.fit(model, data_module)

    # Test model
    print("\n🧪 Running final evaluation...")
    test_results = trainer.test(model, data_module, ckpt_path='best')

    # Track and save metrics
    metrics_tracker = MetricsTracker("results")
    metrics_df = metrics_tracker.extract_metrics(trainer)
    
    if not metrics_df.empty:
        metrics_df.to_csv("results/training_metrics.csv", index=False)
        print("📊 Metrics saved to results/training_metrics.csv")
        
        try:
            metrics_tracker.plot_metrics(metrics_df)
            print("📈 Training plots saved to results/training_metrics.png")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print("\n✅ Training completed successfully!")
    print(f"🏆 Best model: {checkpoint_callback.best_model_path}")

    # Print final test results
    print("\n📋 **FINAL TEST RESULTS:**")
    print("-"*50)
    for key, value in trainer.logged_metrics.items():
        if key.startswith('test_'):
            print(f"{key:<25}: {value:.4f}")
    
    # ✅ Print learned loss weights
    print("\n📊 **LEARNED LOSS WEIGHTS:**")
    print("-"*50)
    if hasattr(model.model, 'log_vars'):
        print(f"Regression weight:      {torch.exp(-model.model.log_vars[0]).item():.4f}")
        print(f"Classification weight:  {torch.exp(-model.model.log_vars[1]).item():.4f}")
        print(f"Ranking weight:         {torch.exp(-model.model.log_vars[2]).item():.4f}")


if __name__ == '__main__':
    main()

