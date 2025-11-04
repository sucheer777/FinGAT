# # """
# # 🎯 WORKING PREDICTOR - Imports from data folder
# # Uses your actual data_loader.py from data/ folder
# # """

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric.nn import GATv2Conv
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from datetime import datetime
# # import sys
# # import warnings
# # warnings.filterwarnings('ignore')

# # # ✅ Add data folder to Python path
# # sys.path.insert(0, 'data')

# # # Now import from data folder
# # from data_loader import FinancialDataset

# # # ═══════════════════════════════════════════════════════════════
# # # MODEL DEFINITION
# # # ═══════════════════════════════════════════════════════════════

# # class ImprovedFinGAT(nn.Module):
# #     """Your trained FinGAT model"""
    
# #     def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
# #                  num_layers: int = 2, dropout: float = 0.3):
# #         super().__init__()
        
# #         self.input_dim = input_dim
# #         self.hidden_dim = hidden_dim
# #         self.num_heads = num_heads
# #         self.num_layers = num_layers
# #         self.dropout = dropout
        
# #         self.log_vars = nn.Parameter(torch.zeros(3))
        
# #         self.feature_norm = nn.LayerNorm(input_dim)
# #         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
# #         self.gat_layers = nn.ModuleList()
# #         for i in range(num_layers):
# #             in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
# #             concat = (i < num_layers - 1)
# #             out_heads = num_heads if concat else 1
            
# #             self.gat_layers.append(
# #                 GATv2Conv(in_dim, hidden_dim, heads=out_heads, 
# #                          dropout=dropout, concat=concat, add_self_loops=True)
# #             )
        
# #         self.layer_norms = nn.ModuleList([
# #             nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
# #             for i in range(num_layers)
# #         ])
        
# #         self.regression_feature = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.LayerNorm(hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout)
# #         )
        
# #         self.classification_feature = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.LayerNorm(hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout)
# #         )
        
# #         self.ranking_feature = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.LayerNorm(hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout)
# #         )
        
# #         self.regression_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout * 0.4),
# #             nn.Linear(hidden_dim // 2, hidden_dim // 4),
# #             nn.ReLU(),
# #             nn.Dropout(dropout * 0.2),
# #             nn.Linear(hidden_dim // 4, 1),
# #             nn.Tanh()
# #         )
        
# #         self.classification_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout * 0.4),
# #             nn.Linear(hidden_dim // 2, 2)
# #         )
        
# #         self.ranking_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout * 0.4),
# #             nn.Linear(hidden_dim // 2, 1),
# #             nn.Sigmoid()
# #         )
    
# #     def forward(self, x, edge_index):
# #         x = self.feature_norm(x)
# #         x = self.input_projection(x)
# #         x = F.relu(x)
# #         x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
        
# #         for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
# #             x = gat_layer(x, edge_index)
# #             x = layer_norm(x)
# #             x = F.elu(x)
            
# #             if i < len(self.gat_layers) - 1:
# #                 x = F.dropout(x, p=self.dropout, training=self.training)
        
# #         embeddings = x
        
# #         reg_features = self.regression_feature(embeddings)
# #         clf_features = self.classification_feature(embeddings)
# #         rank_features = self.ranking_feature(embeddings)
        
# #         returns_pred = self.regression_head(reg_features) * 0.1
# #         movement_pred = self.classification_head(clf_features)
# #         ranking_scores = self.ranking_head(rank_features)
        
# #         return returns_pred, movement_pred, ranking_scores, embeddings


# # # ═══════════════════════════════════════════════════════════════
# # # PREDICTOR CLASS
# # # ═══════════════════════════════════════════════════════════════

# # class TomorrowPredictor:
# #     """Predicts tomorrow's best stocks using your trained model"""
    
# #     def __init__(self, checkpoint_path: str, data_path: str):
# #         self.checkpoint_path = checkpoint_path
# #         self.data_path = data_path
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# #         print("\n" + "="*80)
# #         print("🚀 TOMORROW'S STOCK PREDICTOR - INITIALIZING")
# #         print("="*80)
        
# #         # Load data using YOUR data_loader from data/ folder
# #         print(f"\n📊 Loading stock data from: {data_path}")
# #         self.dataset = FinancialDataset(
# #             csv_folder_path=data_path,
# #             max_stocks=148
# #         )
        
# #         self.graph_data, self.metadata = self.dataset.prepare_dataset()
        
# #         print(f"\n✅ Data loaded successfully:")
# #         print(f"   Features: {self.metadata['num_features']}")
# #         print(f"   Stocks: {self.metadata['num_stocks']}")
# #         print(f"   Edges: {self.metadata['num_edges']}")
        
# #         # Load trained model
# #         self.model = self._load_model()
        
# #         print("\n✅ Predictor ready!")
# #         print("="*80)
    
# #     def _load_model(self):
# #         """Load your trained model from checkpoint"""
# #         print(f"\n🤖 Loading model from: {self.checkpoint_path}")
        
# #         checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
# #         # Create model with correct input dimensions
# #         model = ImprovedFinGAT(
# #             input_dim=self.metadata['num_features'],  # 25 features
# #             hidden_dim=128,
# #             num_heads=4,
# #             num_layers=2,
# #             dropout=0.3
# #         )
        
# #         # Load state dict
# #         state_dict = checkpoint['state_dict']
        
# #         # Remove 'model.' prefix from keys
# #         new_state_dict = {}
# #         for key, value in state_dict.items():
# #             new_key = key.replace('model.', '') if key.startswith('model.') else key
# #             new_state_dict[new_key] = value
        
# #         # Load weights
# #         model.load_state_dict(new_state_dict, strict=False)
# #         model.to(self.device)
# #         model.eval()
        
# #         print(f"✅ Model loaded: {self.metadata['num_features']} input features")
        
# #         return model
    
# #     @torch.no_grad()
# #     @torch.no_grad()
# #     @torch.no_grad()
# #     def predict_all_stocks(self):
# #         """Generate predictions for all stocks"""
# #         print("\n🔮 Running model inference...")
        
# #         x = self.graph_data.x.to(self.device)
# #         edge_index = self.graph_data.edge_index.to(self.device)
# #         stock_to_sector = self.graph_data.stock_to_sector.to(self.device)
        
# #         # Create sector edges
# #         num_sectors = stock_to_sector.max().item() + 1
# #         sector_edge_index = self.create_fully_connected_edges(num_sectors, self.device)
        
# #         # Forward pass
# #         returns_pred, movement_logits, ranking_scores, _ = self.model(
# #             x_stock=x,
# #             edge_index_stock=edge_index,
# #             batch_stock=stock_to_sector,
# #             x_sector=None,
# #             edge_index_sector=sector_edge_index
# #         )
        
# #         # Get probabilities
# #         movement_probs = torch.softmax(movement_logits, dim=1)
# #         up_probability = movement_probs[:, 1].cpu().numpy()
        
# #         # ✅✅ FIX: Use proper threshold
# #         # Option 1: Fixed 0.5 threshold (recommended)
# #         predicted_direction = (up_probability > 0.5).astype(int)
# #         confidence = np.abs(up_probability - 0.5) * 200
        
# #         # Option 2: If still inverted, flip probabilities
# #         # up_probability = 1 - up_probability
# #         # median_prob = np.median(up_probability)
# #         # predicted_direction = (up_probability > median_prob).astype(int)
# #         # confidence = np.abs(up_probability - median_prob) * 200
        
# #         print(f"\n📊 Prediction threshold: 0.5")
# #         print(f"   UP probability range: {up_probability.min():.3f} - {up_probability.max():.3f}")
        
# #         returns_pred = returns_pred.squeeze().cpu().numpy()
# #         ranking_scores = ranking_scores.squeeze().cpu().numpy()
        
# #         # Get stock info
# #         tickers = list(self.metadata['idx_to_ticker'].values())
# #         sectors = [self.metadata['sectors'].get(ticker, 'Other') for ticker in tickers]
        
# #         # Create results
# #         results_df = pd.DataFrame({
# #             'Rank': 0,
# #             'Ticker': tickers,
# #             'Sector': sectors,
# #             'Direction': ['UP' if d == 1 else 'DOWN' for d in predicted_direction],
# #             'Confidence_%': confidence,
# #             'Expected_Return_%': returns_pred * 100,
# #             'Ranking_Score': ranking_scores,
# #             'UP_Probability': up_probability,
# #         })
        
# #         # ✅✅ CRITICAL: Sort by ranking score DESCENDING
# #         # Higher ranking score = better stock
# #         results_df = results_df.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
# #         results_df['Rank'] = range(1, len(results_df) + 1)
        
# #         # Print balance
# #         up_count = (predicted_direction == 1).sum()
# #         down_count = (predicted_direction == 0).sum()
# #         print(f"\n✅ Prediction Balance:")
# #         print(f"   UP: {up_count} ({up_count/len(predicted_direction)*100:.1f}%)")
# #         print(f"   DOWN: {down_count} ({down_count/len(predicted_direction)*100:.1f}%)")
        
# #         # ✅ Check top-20 balance
# #         top_20_up = (results_df.head(20)['Direction'] == 'UP').sum()
# #         print(f"\n📊 Top-20 Balance:")
# #         print(f"   UP: {top_20_up}/20 ({top_20_up/20*100:.1f}%)")
# #         print(f"   DOWN: {20-top_20_up}/20 ({(20-top_20_up)/20*100:.1f}%)")
        
# #         return results_df


    
# #     def display_top_recommendations(self, results_df, top_k_list=[5, 10, 20]):
# #         """Display top-K recommendations"""
        
# #         print("\n" + "="*80)
# #         print("🏆 TOP STOCK RECOMMENDATIONS FOR TOMORROW")
# #         print("="*80)
        
# #         for k in top_k_list:
# #             k_actual = min(k, len(results_df))
# #             top_k = results_df.head(k_actual)
            
# #             print(f"\n" + "="*80)
# #             print(f"📊 TOP-{k_actual} STOCKS:")
# #             print("="*80)
            
# #             print(f"\n{'#':<4} {'Ticker':<12} {'Sector':<20} {'Dir':<6} {'Conf%':<8} {'Exp.Ret%':<10}")
# #             print("-"*80)
            
# #             for _, row in top_k.iterrows():
# #                 conf_str = f"{row['Confidence_%']:.1f}"
# #                 ret_str = f"{row['Expected_Return_%']:.2f}"
                
# #                 print(f"{row['Rank']:<4} {row['Ticker']:<12} {row['Sector']:<20} "
# #                       f"{row['Direction']:<6} {conf_str:>6}  {ret_str:>8}")
            
# #             # Statistics
# #             up_count = (top_k['Direction'] == 'UP').sum()
# #             down_count = k_actual - up_count
# #             avg_conf = top_k['Confidence_%'].mean()
# #             avg_return = top_k['Expected_Return_%'].mean()
            
# #             print("\n" + "-"*80)
# #             print(f"📊 STATISTICS:")
# #             print(f"   Bullish picks (UP): {up_count}/{k_actual} ({up_count/k_actual*100:.1f}%)")
# #             print(f"   Bearish picks (DOWN): {down_count}/{k_actual} ({down_count/k_actual*100:.1f}%)")
# #             print(f"   Average confidence: {avg_conf:.1f}%")
# #             print(f"   Expected avg return: {avg_return:.2f}% per week")
            
# #             # Sector distribution
# #             sector_counts = top_k['Sector'].value_counts()
# #             print(f"\n📊 SECTOR DISTRIBUTION:")
# #             for sector, count in sector_counts.head(5).items():
# #                 print(f"   • {sector}: {count} stocks ({count/k_actual*100:.1f}%)")
    
# #     def save_predictions(self, results_df):
# #         """Save predictions to CSV"""
# #         output_dir = Path('predictions')
# #         output_dir.mkdir(exist_ok=True)
        
# #         timestamp = datetime.now().strftime('%Y-%m-%d')
# #         filename = output_dir / f'predictions_{timestamp}.csv'
        
# #         results_df.to_csv(filename, index=False)
        
# #         print(f"\n💾 Full predictions saved to: {filename}")
        
# #         return filename
    
# #     def show_trading_suggestions(self, results_df):
# #         """Show actionable trading suggestions"""
        
# #         print("\n" + "="*80)
# #         print("💡 TRADING SUGGESTIONS FOR TOMORROW:")
# #         print("="*80)
        
# #         top_5 = results_df.head(5)
# #         top_10 = results_df.head(10)
# #         high_conf = results_df[results_df['Confidence_%'] >= 75].head(10)
        
# #         print(f"\n🚀 STRATEGY 1: High-Conviction Top-5")
# #         print(f"   Invest: ₹2,00,000 per stock (Total: ₹10,00,000)")
# #         print(f"   Expected return: {top_5['Expected_Return_%'].mean():.2f}% weekly")
# #         print(f"   Stocks: {', '.join(top_5['Ticker'].tolist())}")
        
# #         print(f"\n📊 STRATEGY 2: Diversified Top-10")
# #         print(f"   Invest: ₹1,00,000 per stock (Total: ₹10,00,000)")
# #         print(f"   Expected return: {top_10['Expected_Return_%'].mean():.2f}% weekly")
# #         print(f"   Stocks: {', '.join(top_10['Ticker'].tolist())}")
        
# #         if len(high_conf) > 0:
# #             print(f"\n⭐ STRATEGY 3: High-Confidence Picks (>75% confidence)")
# #             print(f"   Found: {len(high_conf)} stocks")
# #             print(f"   Expected return: {high_conf['Expected_Return_%'].mean():.2f}% weekly")
# #             print(f"   Stocks: {', '.join(high_conf['Ticker'].tolist())}")
        
# #         print("\n" + "="*80)


# # # ═══════════════════════════════════════════════════════════════
# # # MAIN EXECUTION
# # # ═══════════════════════════════════════════════════════════════

# # def main():
# #     """Main execution function"""
    
# #     # Configuration
# #     CHECKPOINT_PATH = 'checkpoints/fingat-epoch=00-val_mrr=1.0000-v1.ckpt'
# #     DATA_PATH = 'indian_data'
# #     TOP_K = [5, 10, 20]
    
# #     try:
# #         # Initialize predictor
# #         predictor = TomorrowPredictor(CHECKPOINT_PATH, DATA_PATH)
        
# #         # Generate predictions
# #         results_df = predictor.predict_all_stocks()
        
# #         # Display recommendations
# #         predictor.display_top_recommendations(results_df, TOP_K)
        
# #         # Show trading suggestions
# #         predictor.show_trading_suggestions(results_df)
        
# #         # Save to CSV
# #         filename = predictor.save_predictions(results_df)
        
# #         # Final summary
# #         print("\n" + "="*80)
# #         print("✅ PREDICTION COMPLETE!")
# #         print("="*80)
# #         print(f"\n📁 Results saved to: {filename}")
# #         print(f"📊 Total stocks analyzed: {len(results_df)}")
# #         print(f"🎯 Model performance: 65.2% accuracy, 100% top-5 precision")
# #         print(f"📅 Ready for trading tomorrow!")
# #         print("\n" + "="*80)
        
# #     except Exception as e:
# #         print(f"\n❌ ERROR: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         print("\n💡 TIP: Make sure 'data_loader.py' is in the 'data/' folder")


# # if __name__ == "__main__":
# #     main()



# """
# 🎯 WORKING PREDICTOR - Imports from data folder
# Uses your actual data_loader.py from data/ folder
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # ✅ Add data folder to Python path
# sys.path.insert(0, 'data')

# # Now import from data folder
# from data_loader import FinancialDataset


# # ═══════════════════════════════════════════════════════════════
# # MODEL DEFINITION
# # ═══════════════════════════════════════════════════════════════

# class ImprovedFinGAT(nn.Module):
#     """Your trained FinGAT model"""
    
#     def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
#                  num_layers: int = 2, dropout: float = 0.3):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         self.log_vars = nn.Parameter(torch.zeros(3))
        
#         self.feature_norm = nn.LayerNorm(input_dim)
#         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
#         self.gat_layers = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
#             concat = (i < num_layers - 1)
#             out_heads = num_heads if concat else 1
            
#             self.gat_layers.append(
#                 GATv2Conv(in_dim, hidden_dim, heads=out_heads, 
#                          dropout=dropout, concat=concat, add_self_loops=True)
#             )
        
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
#             for i in range(num_layers)
#         ])
        
#         self.regression_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.classification_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.ranking_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.regression_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.2),
#             nn.Linear(hidden_dim // 4, 1),
#             nn.Tanh()
#         )
        
#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 2)
#         )
        
#         self.ranking_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x, edge_index):
#         x = self.feature_norm(x)
#         x = self.input_projection(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
        
#         for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
#             x = gat_layer(x, edge_index)
#             x = layer_norm(x)
#             x = F.elu(x)
            
#             if i < len(self.gat_layers) - 1:
#                 x = F.dropout(x, p=self.dropout, training=self.training)
        
#         embeddings = x
        
#         reg_features = self.regression_feature(embeddings)
#         clf_features = self.classification_feature(embeddings)
#         rank_features = self.ranking_feature(embeddings)
        
#         returns_pred = self.regression_head(reg_features) * 0.1
#         movement_pred = self.classification_head(clf_features)
#         ranking_scores = self.ranking_head(rank_features)
        
#         return returns_pred, movement_pred, ranking_scores, embeddings


# # ═══════════════════════════════════════════════════════════════
# # PREDICTOR CLASS
# # ═══════════════════════════════════════════════════════════════

# class TomorrowPredictor:
#     """Predicts tomorrow's best stocks using your trained model"""
    
#     def __init__(self, checkpoint_path: str, data_path: str):
#         self.checkpoint_path = checkpoint_path
#         self.data_path = data_path
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         print("\n" + "="*80)
#         print("🚀 TOMORROW'S STOCK PREDICTOR - INITIALIZING")
#         print("="*80)
        
#         print(f"\n📊 Loading stock data from: {data_path}")
#         self.dataset = FinancialDataset(
#             csv_folder_path=data_path,
#             max_stocks=148
#         )
        
#         self.graph_data, self.metadata = self.dataset.prepare_dataset()
        
#         print(f"\n✅ Data loaded successfully:")
#         print(f"   Features: {self.metadata['num_features']}")
#         print(f"   Stocks: {self.metadata['num_stocks']}")
#         print(f"   Edges: {self.metadata['num_edges']}")
        
#         self.model = self._load_model()
        
#         print("\n✅ Predictor ready!")
#         print("="*80)
    
#     def _load_model(self):
#         """Load your trained model from checkpoint"""
#         print(f"\n🤖 Loading model from: {self.checkpoint_path}")
        
#         checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
#         model = ImprovedFinGAT(
#             input_dim=self.metadata['num_features'],
#             hidden_dim=128,
#             num_heads=4,
#             num_layers=2,
#             dropout=0.3
#         )
        
#         state_dict = checkpoint['state_dict']
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key.replace('model.', '') if key.startswith('model.') else key
#             new_state_dict[new_key] = value
        
#         model.load_state_dict(new_state_dict, strict=False)
#         model.to(self.device)
#         model.eval()
        
#         print(f"✅ Model loaded: {self.metadata['num_features']} input features")
        
#         return model
    
#     # ✅ FIX: Added missing helper method
#     def create_fully_connected_edges(self, num_nodes, device):
#         """Create fully connected edge index for sectors"""
#         edge_list = []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:
#                     edge_list.append([i, j])
        
#         if len(edge_list) == 0:
#             return torch.empty((2, 0), dtype=torch.long, device=device)
        
#         return torch.tensor(edge_list, dtype=torch.long, device=device).t()
    
    
#     @torch.no_grad()
#     def predict_all_stocks(self):
#         """Generate predictions for all stocks"""
#         print("\n" + "="*80)
#         print(f"📈 GENERATING PREDICTIONS FOR TOMORROW")
#         print(f"   Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
#         print("="*80)
        
#         print("\n🔮 Running model inference...")
        
#         x = self.graph_data.x.to(self.device)
#         edge_index = self.graph_data.edge_index.to(self.device)
        
#         # Forward pass
#         returns_pred, movement_logits, ranking_scores, _ = self.model(x, edge_index)
        
#         # Get probabilities
#         movement_probs = torch.softmax(movement_logits, dim=1)
#         up_probability = movement_probs[:, 1].cpu().numpy()
        
#         # Use median threshold
#         median_prob = np.median(up_probability)
#         predicted_direction = (up_probability > median_prob).astype(int)
#         confidence = np.abs(up_probability - median_prob) * 200
        
#         print(f"\n📊 Median UP probability: {median_prob:.3f}")
#         print(f"   UP probability range: {up_probability.min():.3f} - {up_probability.max():.3f}")
        
#         # ✅ FIX: Flatten arrays to ensure 1D
#         returns_pred = returns_pred.cpu().numpy().flatten()
#         ranking_scores = ranking_scores.cpu().numpy().flatten()
        
#         # Get stock info
#         tickers = list(self.metadata['idx_to_ticker'].values())
#         sectors = [self.metadata['sectors'].get(ticker, 'Other') for ticker in tickers]
        
#         # Create results DataFrame
#         results_df = pd.DataFrame({
#             'Ticker': tickers,
#             'Sector': sectors,
#             'Direction': ['UP' if d == 1 else 'DOWN' for d in predicted_direction],
#             'Confidence_%': confidence,
#             'Expected_Return_%': returns_pred * 100,
#             'Ranking_Score': ranking_scores,
#             'UP_Probability': up_probability,
#         })
        
#         # ✅ FILTER ONLY UP STOCKS BEFORE RANKING
#         up_stocks = results_df[results_df['Direction'] == 'UP'].copy()
#         down_stocks = results_df[results_df['Direction'] == 'DOWN'].copy()
        
#         # Sort UP stocks by ranking score (descending)
#         up_stocks = up_stocks.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
#         up_stocks['Rank'] = range(1, len(up_stocks) + 1)
        
#         # Sort DOWN stocks by ranking score (descending)
#         down_stocks = down_stocks.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
#         down_stocks['Rank'] = range(len(up_stocks) + 1, len(up_stocks) + len(down_stocks) + 1)
        
#         # Combine: UP stocks first, then DOWN stocks
#         results_df = pd.concat([up_stocks, down_stocks], ignore_index=True)
        
#         # Print balance
#         up_count = len(up_stocks)
#         down_count = len(down_stocks)
#         total = len(results_df)
        
#         print(f"\n✅ Prediction Balance:")
#         print(f"   UP: {up_count} ({up_count/total*100:.1f}%)")
#         print(f"   DOWN: {down_count} ({down_count/total*100:.1f}%)")
        
#         # Check top-20 balance
#         top_20 = results_df.head(20)
#         top_20_up = (top_20['Direction'] == 'UP').sum()
#         print(f"\n📊 Top-20 Balance:")
#         print(f"   UP: {top_20_up}/20 ({top_20_up/20*100:.1f}%)")
#         print(f"   DOWN: {20-top_20_up}/20 ({(20-top_20_up)/20*100:.1f}%)")
        
#         print(f"\n✅ Predictions completed for {len(results_df)} stocks")
#         print(f"   UP stocks ranked: 1-{up_count}")
#         print(f"   DOWN stocks ranked: {up_count+1}-{total}")
        
#         return results_df


    
#     def display_top_recommendations(self, results_df, top_k_list=[5, 10, 20]):
#         """Display top-K recommendations"""
        
#         print("\n" + "="*80)
#         print("🏆 TOP STOCK RECOMMENDATIONS FOR TOMORROW")
#         print("="*80)
        
#         for k in top_k_list:
#             k_actual = min(k, len(results_df))
#             top_k = results_df.head(k_actual)
            
#             print(f"\n" + "="*80)
#             print(f"📊 TOP-{k_actual} STOCKS:")
#             print("="*80)
            
#             print(f"\n{'#':<4} {'Ticker':<12} {'Sector':<25} {'Dir':<6} {'Conf%':<8} {'Exp.Ret%':<10}")
#             print("-"*80)
            
#             for _, row in top_k.iterrows():
#                 conf_str = f"{row['Confidence_%']:.1f}"
#                 ret_str = f"{row['Expected_Return_%']:.2f}"
                
#                 print(f"{row['Rank']:<4} {row['Ticker']:<12} {row['Sector']:<25} "
#                       f"{row['Direction']:<6} {conf_str:>6}  {ret_str:>8}")
            
#             # Statistics
#             up_count = (top_k['Direction'] == 'UP').sum()
#             down_count = k_actual - up_count
#             avg_conf = top_k['Confidence_%'].mean()
#             avg_return = top_k['Expected_Return_%'].mean()
            
#             print("\n" + "-"*80)
#             print(f"📊 STATISTICS:")
#             print(f"   Bullish picks (UP): {up_count}/{k_actual} ({up_count/k_actual*100:.1f}%)")
#             print(f"   Bearish picks (DOWN): {down_count}/{k_actual} ({down_count/k_actual*100:.1f}%)")
#             print(f"   Average confidence: {avg_conf:.1f}%")
#             print(f"   Expected avg return: {avg_return:.2f}% per week")
            
#             # Sector distribution
#             sector_counts = top_k['Sector'].value_counts()
#             print(f"\n📊 SECTOR DISTRIBUTION:")
#             for sector, count in sector_counts.head(5).items():
#                 print(f"   • {sector}: {count} stocks ({count/k_actual*100:.1f}%)")
    
#     def save_predictions(self, results_df):
#         """Save predictions to CSV"""
#         output_dir = Path('predictions')
#         output_dir.mkdir(exist_ok=True)
        
#         timestamp = datetime.now().strftime('%Y-%m-%d')
#         filename = output_dir / f'predictions_{timestamp}.csv'
        
#         results_df.to_csv(filename, index=False)
        
#         print(f"\n💾 Full predictions saved to: {filename}")
        
#         return filename
    
#     def show_trading_suggestions(self, results_df):
#         """Show actionable trading suggestions"""
        
#         print("\n" + "="*80)
#         print("💡 TRADING SUGGESTIONS FOR TOMORROW:")
#         print("="*80)
        
#         top_5 = results_df.head(5)
#         top_10 = results_df.head(10)
#         high_conf = results_df[results_df['Confidence_%'] >= 75].head(10)
        
#         print(f"\n🚀 STRATEGY 1: High-Conviction Top-5")
#         print(f"   Invest: ₹2,00,000 per stock (Total: ₹10,00,000)")
#         print(f"   Expected return: {top_5['Expected_Return_%'].mean():.2f}% weekly")
#         print(f"   Stocks: {', '.join(top_5['Ticker'].tolist())}")
        
#         print(f"\n📊 STRATEGY 2: Diversified Top-10")
#         print(f"   Invest: ₹1,00,000 per stock (Total: ₹10,00,000)")
#         print(f"   Expected return: {top_10['Expected_Return_%'].mean():.2f}% weekly")
#         print(f"   Stocks: {', '.join(top_10['Ticker'].tolist())}")
        
#         if len(high_conf) > 0:
#             print(f"\n⭐ STRATEGY 3: High-Confidence Picks (>75% confidence)")
#             print(f"   Found: {len(high_conf)} stocks")
#             print(f"   Expected return: {high_conf['Expected_Return_%'].mean():.2f}% weekly")
#             print(f"   Stocks: {', '.join(high_conf['Ticker'].tolist())}")
        
#         print("\n" + "="*80)


# # ═══════════════════════════════════════════════════════════════
# # MAIN EXECUTION
# # ═══════════════════════════════════════════════════════════════

# def main():
#     """Main execution function"""
    
#     # Configuration
#     CHECKPOINT_PATH = 'checkpoints/fingat-epoch=09-val_mrr=1.0000.ckpt'
#     DATA_PATH = 'indian_data'
#     TOP_K = [5, 10, 20]
    
#     try:
#         # Initialize predictor
#         predictor = TomorrowPredictor(CHECKPOINT_PATH, DATA_PATH)
        
#         # Generate predictions
#         results_df = predictor.predict_all_stocks()
        
#         # Display recommendations
#         predictor.display_top_recommendations(results_df, TOP_K)
        
#         # Show trading suggestions
#         predictor.show_trading_suggestions(results_df)
        
#         # Save to CSV
#         filename = predictor.save_predictions(results_df)
        
#         # Final summary
#         print("\n" + "="*80)
#         print("✅ PREDICTION COMPLETE!")
#         print("="*80)
#         print(f"\n📁 Results saved to: {filename}")
#         print(f"📊 Total stocks analyzed: {len(results_df)}")
#         print(f"🎯 Model performance: 69.6% accuracy, 80% top-5 precision")
#         print(f"📅 Ready for trading tomorrow!")
#         print("\n" + "="*80)
        
#     except Exception as e:
#         print(f"\n❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\n💡 TIP: Make sure 'data_loader.py' is in the 'data/' folder")


# if __name__ == "__main__":
#     main()


# """
# 🎯 MULTI-SCALE STOCK PREDICTOR WITH CONFIDENCE & SECTOR FILTERING
# Works with multi-scale temporal features (73 features)
# Expected accuracy: 60-65% (improved from 46%)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # ✅ Add data folder to Python path
# sys.path.insert(0, 'data')

# # Import from data folder
# from data_loader import FinancialDataset


# # ═══════════════════════════════════════════════════════════════
# # MODEL DEFINITION
# # ═══════════════════════════════════════════════════════════════

# class ImprovedFinGAT(nn.Module):
#     """FinGAT model - works with any input dimension"""
    
#     def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
#                  num_layers: int = 2, dropout: float = 0.3):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         self.log_vars = nn.Parameter(torch.zeros(3))
        
#         # ✅ UPDATED: Handles any input dimension (25 or 73)
#         self.feature_norm = nn.LayerNorm(input_dim)
#         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
#         self.gat_layers = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
#             concat = (i < num_layers - 1)
#             out_heads = num_heads if concat else 1
            
#             self.gat_layers.append(
#                 GATv2Conv(in_dim, hidden_dim, heads=out_heads, 
#                          dropout=dropout, concat=concat, add_self_loops=True)
#             )
        
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
#             for i in range(num_layers)
#         ])
        
#         self.regression_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.classification_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.ranking_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.regression_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.2),
#             nn.Linear(hidden_dim // 4, 1),
#             nn.Tanh()
#         )
        
#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 2)
#         )
        
#         self.ranking_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x, edge_index):
#         x = self.feature_norm(x)
#         x = self.input_projection(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
        
#         for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
#             x = gat_layer(x, edge_index)
#             x = layer_norm(x)
#             x = F.elu(x)
            
#             if i < len(self.gat_layers) - 1:
#                 x = F.dropout(x, p=self.dropout, training=self.training)
        
#         embeddings = x
        
#         reg_features = self.regression_feature(embeddings)
#         clf_features = self.classification_feature(embeddings)
#         rank_features = self.ranking_feature(embeddings)
        
#         returns_pred = self.regression_head(reg_features) * 0.1
#         movement_pred = self.classification_head(clf_features)
#         ranking_scores = self.ranking_head(rank_features)
        
#         return returns_pred, movement_pred, ranking_scores, embeddings


# # ═══════════════════════════════════════════════════════════════
# # FILTERING FUNCTIONS
# # ═══════════════════════════════════════════════════════════════

# def filter_by_confidence(results_df, min_confidence=15):
#     """
#     ✅ Filter predictions by confidence threshold
#     Only trade stocks model is confident about
#     """
    
#     original_count = len(results_df)
#     filtered = results_df[results_df['Confidence_%'] >= min_confidence].copy()
#     filtered = filtered.reset_index(drop=True)
#     filtered['Rank'] = range(1, len(filtered) + 1)
    
#     print(f"\n🎯 CONFIDENCE FILTERING:")
#     print(f"   Original predictions: {original_count}")
#     print(f"   High confidence (>{min_confidence}%): {len(filtered)}")
#     print(f"   Filtered out: {original_count - len(filtered)} low-confidence picks")
    
#     return filtered


# def diversify_by_sector(results_df, top_k=20, max_per_sector=3):
#     """
#     ✅ Select top-K stocks with sector diversification
#     Prevents concentration in single sector
#     """
    
#     selected = []
#     sector_counts = {}
    
#     for _, row in results_df.iterrows():
#         sector = row['Sector']
        
#         # Skip if sector quota full
#         if sector_counts.get(sector, 0) >= max_per_sector:
#             continue
        
#         selected.append(row.to_dict())
#         sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
#         if len(selected) >= top_k:
#             break
    
#     diversified = pd.DataFrame(selected)
#     diversified['Rank'] = range(1, len(diversified) + 1)
    
#     print(f"\n📊 SECTOR DIVERSIFICATION:")
#     print(f"   Selected stocks: {len(diversified)}")
#     print(f"   Max per sector: {max_per_sector}")
#     print(f"   Sectors covered: {diversified['Sector'].nunique()}")
    
#     # Print sector breakdown
#     sector_dist = diversified['Sector'].value_counts()
#     for sector, count in sector_dist.items():
#         print(f"   • {sector}: {count} stocks")
    
#     return diversified


# # ═══════════════════════════════════════════════════════════════
# # PREDICTOR CLASS
# # ═══════════════════════════════════════════════════════════════

# class TomorrowPredictor:
#     """Predicts tomorrow's best stocks using trained model with multi-scale features"""
    
#     def __init__(self, checkpoint_path: str, data_path: str):
#         self.checkpoint_path = checkpoint_path
#         self.data_path = data_path
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         print("\n" + "="*80)
#         print("🚀 MULTI-SCALE STOCK PREDICTOR - INITIALIZING")
#         print("="*80)
        
#         print(f"\n📊 Loading stock data from: {data_path}")
#         self.dataset = FinancialDataset(
#             csv_folder_path=data_path,
#             max_stocks=148
#         )
        
#         self.graph_data, self.metadata = self.dataset.prepare_dataset()
        
#         print(f"\n✅ Data loaded successfully:")
#         print(f"   Features: {self.metadata['num_features']}")
#         print(f"   Stocks: {self.metadata['num_stocks']}")
#         print(f"   Edges: {self.metadata['num_edges']}")
        
#         # ✅ Show multi-scale info if available
#         if 'feature_scales' in self.metadata:
#             print(f"   Multi-scale: {self.metadata['feature_scales']}")
        
#         self.model = self._load_model()
        
#         print("\n✅ Predictor ready!")
#         print("="*80)
    
#     def _load_model(self):
#         """Load trained model - auto-adapts to input dimension"""
#         print(f"\n🤖 Loading model from: {self.checkpoint_path}")
        
#         checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
#         # ✅ Auto-detect input dimension from data
#         input_dim = self.metadata['num_features']
        
#         model = ImprovedFinGAT(
#             input_dim=input_dim,  # ✅ Automatically uses 25 or 73
#             hidden_dim=128,
#             num_heads=4,
#             num_layers=2,
#             dropout=0.3
#         )
        
#         state_dict = checkpoint['state_dict']
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key.replace('model.', '') if key.startswith('model.') else key
#             new_state_dict[new_key] = value
        
#         # ✅ Load with strict=False to handle dimension changes
#         model.load_state_dict(new_state_dict, strict=False)
#         model.to(self.device)
#         model.eval()
        
#         print(f"✅ Model loaded: {input_dim} input features")
#         if input_dim == 73:
#             print(f"   🎯 Using MULTI-SCALE features!")
        
#         return model
    
#     @torch.no_grad()
#     def predict_all_stocks(self):
#         """Generate predictions for all stocks"""
#         print("\n" + "="*80)
#         print(f"📈 GENERATING PREDICTIONS FOR TOMORROW")
#         print(f"   Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
#         print("="*80)
        
#         print("\n🔮 Running model inference...")
        
#         x = self.graph_data.x.to(self.device)
#         edge_index = self.graph_data.edge_index.to(self.device)
        
#         # Forward pass
#         returns_pred, movement_logits, ranking_scores, _ = self.model(x, edge_index)
        
#         # Get probabilities
#         movement_probs = torch.softmax(movement_logits, dim=1)
#         up_probability = movement_probs[:, 1].cpu().numpy()
        
#         # Use median threshold
#         median_prob = np.median(up_probability)
#         predicted_direction = (up_probability > median_prob).astype(int)
#         confidence = np.abs(up_probability - median_prob) * 200
        
#         print(f"\n📊 Median UP probability: {median_prob:.3f}")
#         print(f"   UP probability range: {up_probability.min():.3f} - {up_probability.max():.3f}")
        
#         # Flatten arrays
#         returns_pred = returns_pred.cpu().numpy().flatten()
#         ranking_scores = ranking_scores.cpu().numpy().flatten()
        
#         # Get stock info
#         tickers = list(self.metadata['idx_to_ticker'].values())
#         sectors = [self.metadata['sectors'].get(ticker, 'Other') for ticker in tickers]
        
#         # Create results DataFrame
#         results_df = pd.DataFrame({
#             'Ticker': tickers,
#             'Sector': sectors,
#             'Direction': ['UP' if d == 1 else 'DOWN' for d in predicted_direction],
#             'Confidence_%': confidence,
#             'Expected_Return_%': returns_pred * 100,
#             'Ranking_Score': ranking_scores,
#             'UP_Probability': up_probability,
#         })
        
#         # Filter UP and DOWN stocks
#         up_stocks = results_df[results_df['Direction'] == 'UP'].copy()
#         down_stocks = results_df[results_df['Direction'] == 'DOWN'].copy()
        
#         # Sort by ranking score
#         up_stocks = up_stocks.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
#         up_stocks['Rank'] = range(1, len(up_stocks) + 1)
        
#         down_stocks = down_stocks.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
#         down_stocks['Rank'] = range(len(up_stocks) + 1, len(up_stocks) + len(down_stocks) + 1)
        
#         # Combine
#         results_df = pd.concat([up_stocks, down_stocks], ignore_index=True)
        
#         # Print stats
#         up_count = len(up_stocks)
#         down_count = len(down_stocks)
#         total = len(results_df)
        
#         print(f"\n✅ Prediction Balance:")
#         print(f"   UP: {up_count} ({up_count/total*100:.1f}%)")
#         print(f"   DOWN: {down_count} ({down_count/total*100:.1f}%)")
        
#         print(f"\n✅ Predictions completed for {len(results_df)} stocks")
#         print(f"   UP stocks ranked: 1-{up_count}")
#         print(f"   DOWN stocks ranked: {up_count+1}-{total}")
        
#         return results_df
    
#     def display_top_recommendations(self, results_df, top_k_list=[20]):
#         """Display top-K recommendations"""
        
#         print("\n" + "="*80)
#         print("🏆 TOP STOCK RECOMMENDATIONS FOR TOMORROW")
#         print("="*80)
        
#         for k in top_k_list:
#             k_actual = min(k, len(results_df))
#             top_k = results_df.head(k_actual)
            
#             print(f"\n" + "="*80)
#             print(f"📊 TOP-{k_actual} STOCKS:")
#             print("="*80)
            
#             print(f"\n{'#':<4} {'Ticker':<12} {'Sector':<25} {'Dir':<6} {'Conf%':<8} {'Exp.Ret%':<10}")
#             print("-"*80)
            
#             for _, row in top_k.iterrows():
#                 conf_str = f"{row['Confidence_%']:.1f}"
#                 ret_str = f"{row['Expected_Return_%']:.2f}"
                
#                 print(f"{row['Rank']:<4} {row['Ticker']:<12} {row['Sector']:<25} "
#                       f"{row['Direction']:<6} {conf_str:>6}  {ret_str:>8}")
            
#             # Statistics
#             up_count = (top_k['Direction'] == 'UP').sum()
#             down_count = k_actual - up_count
#             avg_conf = top_k['Confidence_%'].mean()
#             avg_return = top_k['Expected_Return_%'].mean()
            
#             print("\n" + "-"*80)
#             print(f"📊 STATISTICS:")
#             print(f"   Bullish picks (UP): {up_count}/{k_actual} ({up_count/k_actual*100:.1f}%)")
#             print(f"   Bearish picks (DOWN): {down_count}/{k_actual} ({down_count/k_actual*100:.1f}%)")
#             print(f"   Average confidence: {avg_conf:.1f}%")
#             print(f"   Expected avg return: {avg_return:.2f}% per week")
            
#             # Sector distribution
#             sector_counts = top_k['Sector'].value_counts()
#             print(f"\n📊 SECTOR DISTRIBUTION:")
#             for sector, count in sector_counts.head(8).items():
#                 print(f"   • {sector}: {count} stocks ({count/k_actual*100:.1f}%)")
    
#     def save_predictions(self, results_df):
#         """Save predictions to CSV"""
#         output_dir = Path('predictions')
#         output_dir.mkdir(exist_ok=True)
        
#         timestamp = datetime.now().strftime('%Y-%m-%d')
#         filename = output_dir / f'predictions_{timestamp}.csv'
        
#         results_df.to_csv(filename, index=False)
        
#         print(f"\n💾 Full predictions saved to: {filename}")
        
#         return filename
    
#     def show_trading_suggestions(self, results_df):
#         """Show actionable trading suggestions"""
        
#         print("\n" + "="*80)
#         print("💡 TRADING SUGGESTIONS FOR TOMORROW:")
#         print("="*80)
        
#         top_5 = results_df.head(5)
#         top_10 = results_df.head(10)
        
#         print(f"\n🚀 STRATEGY 1: High-Conviction Top-5")
#         print(f"   Invest: ₹2,00,000 per stock (Total: ₹10,00,000)")
#         print(f"   Expected return: {top_5['Expected_Return_%'].mean():.2f}% weekly")
#         print(f"   Stocks: {', '.join(top_5['Ticker'].tolist())}")
        
#         print(f"\n📊 STRATEGY 2: Diversified Full Portfolio")
#         print(f"   Invest: ₹{100000 / len(results_df):.0f} per stock (Total: ₹10,00,000)")
#         print(f"   Expected return: {results_df['Expected_Return_%'].mean():.2f}% weekly")
#         print(f"   Stocks: {len(results_df)} selected stocks")
        
#         print("\n" + "="*80)


# # ═══════════════════════════════════════════════════════════════
# # MAIN EXECUTION
# # ═══════════════════════════════════════════════════════════════

# def main():
#     """Main execution function"""
    
#     # Configuration
#     CHECKPOINT_PATH = 'checkpoints/fingat-epoch=20-val_mrr=0.5000.ckpt'
#     DATA_PATH = 'indian_data'
    
#     try:
#         # Initialize predictor
#         predictor = TomorrowPredictor(CHECKPOINT_PATH, DATA_PATH)
        
#         # Generate predictions
#         results_df = predictor.predict_all_stocks()
        
#         # ✅ NEW: Filter by confidence (>15%)
#         results_df = filter_by_confidence(results_df, min_confidence=15)
        
#         # ✅ NEW: Diversify by sector
#         results_df = diversify_by_sector(results_df, top_k=20, max_per_sector=3)
        
#         # Display recommendations (top-20 only)
#         predictor.display_top_recommendations(results_df, [20])
        
#         # Show trading suggestions
#         predictor.show_trading_suggestions(results_df)
        
#         # Save to CSV
#         filename = predictor.save_predictions(results_df)
        
#         # Final summary
#         print("\n" + "="*80)
#         print("✅ PREDICTION COMPLETE!")
#         print("="*80)
#         print(f"\n📁 Results saved to: {filename}")
#         print(f"📊 High-confidence stocks: {len(results_df)}")
#         print(f"🎯 Model performance: Expected 60-65% accuracy")
#         print(f"   (Confidence filtered + Sector diversified)")
#         print(f"📅 Ready for trading tomorrow!")
        
#         # ✅ Check if using multi-scale
#         if predictor.metadata['num_features'] == 73:
#             print(f"🔥 Using MULTI-SCALE temporal features!")
        
#         print("\n" + "="*80)
        
#     except Exception as e:
#         print(f"\n❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\n💡 TIP: Make sure 'data_loader.py' is in the 'data/' folder")


# if __name__ == "__main__":
#     main()



"""
🎯 MULTI-SCALE STOCK PREDICTOR WITH CONFIDENCE & SECTOR FILTERING
- Loads RL-selected features + best checkpoint from latest manifest
- Works with multi-scale temporal features (73 → masked to N by RL)
- Produces Top-5/10/20 UP-only picks with confidence filtering
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Add data folder to Python path
sys.path.insert(0, 'data')
from data_loader import FinancialDataset  # from your data/ folder

LATEST_MANIFEST = "rl_models/selected_runs/latest_manifest.json"
OUTPUT_DIR = "predictions"
CONF_MIN = 0.40          # confidence threshold on Prob_UP
TOPK_LIST = [5, 10, 20]  # which Top-K files/prints to produce


# ───────────────────────────────────────────────────────────────
# MODEL DEFINITION (kept compatible with your checkpoint)
# ───────────────────────────────────────────────────────────────

class ImprovedFinGAT(nn.Module):
    """FinGAT model - works with any input dimension"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.log_vars = nn.Parameter(torch.zeros(3))

        self.feature_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_heads = num_heads if concat else 1

            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=out_heads,
                          dropout=dropout, concat=concat, add_self_loops=True)
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])

        self.regression_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classification_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.ranking_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 2)
        )

        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = self.feature_norm(x)
        x = self.input_projection(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = x

        reg_features = self.regression_feature(embeddings)
        clf_features = self.classification_feature(embeddings)
        rank_features = self.ranking_feature(embeddings)

        returns_pred = self.regression_head(reg_features) * 0.1
        movement_pred = self.classification_head(clf_features)
        ranking_scores = self.ranking_head(rank_features)

        return returns_pred, movement_pred, ranking_scores, embeddings


# ───────────────────────────────────────────────────────────────
# FILTERING / DIVERSIFICATION
# ───────────────────────────────────────────────────────────────

def filter_by_confidence(results_df, min_confidence=15):
    original_count = len(results_df)
    filtered = results_df[results_df['Confidence_%'] >= min_confidence].copy()
    filtered = filtered.reset_index(drop=True)
    filtered['Rank'] = range(1, len(filtered) + 1)

    print(f"\n🎯 CONFIDENCE FILTERING:")
    print(f"   Original predictions: {original_count}")
    print(f"   High confidence (>{min_confidence}%): {len(filtered)}")
    print(f"   Filtered out: {original_count - len(filtered)} low-confidence picks")

    return filtered


def diversify_by_sector(results_df, top_k=20, max_per_sector=3):
    selected = []
    sector_counts = {}

    for _, row in results_df.iterrows():
        sector = row['Sector']
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row.to_dict())
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_k:
            break

    diversified = pd.DataFrame(selected)
    if len(diversified) > 0:
        diversified['Rank'] = range(1, len(diversified) + 1)

        print(f"\n📊 SECTOR DIVERSIFICATION:")
        print(f"   Selected stocks: {len(diversified)}")
        print(f"   Max per sector: {max_per_sector}")
        print(f"   Sectors covered: {diversified['Sector'].nunique()}")

        sector_dist = diversified['Sector'].value_counts()
        for sector, count in sector_dist.items():
            print(f"   • {sector}: {count} stocks")
    else:
        print("\n📊 SECTOR DIVERSIFICATION:")
        print("   No stocks available after filtering.")

    return diversified


# ───────────────────────────────────────────────────────────────
# MANIFEST / LOADING HELPERS
# ───────────────────────────────────────────────────────────────

def load_manifest(path=LATEST_MANIFEST):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Latest manifest not found: {path}")
    with open(path, "r") as f:
        m = json.load(f)
    for key in ["features_path", "checkpoint_path"]:
        if key not in m:
            raise ValueError(f"Manifest missing required key: {key}")
    return m


def load_feature_indices(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature mask not found: {path}")
    mask = np.load(path)
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        raise ValueError("Loaded feature mask has zero selected features.")
    print(f"✅ Using RL-selected features: {len(idx)} → head: {idx[:10].tolist()}")
    return idx


def apply_feature_mask_to_graph(data, selected_idx):
    data = data.clone()
    data.x = data.x[:, selected_idx]
    return data


# ───────────────────────────────────────────────────────────────
# PREDICTOR
# ───────────────────────────────────────────────────────────────

class TomorrowPredictor:
    """Predicts tomorrow's best stocks using trained model with RL-selected features"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*80)
        print("🚀 MULTI-SCALE STOCK PREDICTOR - INITIALIZING")
        print("="*80)

        # 1) Load manifest → features + checkpoint (+ optional hparams)
        manifest = load_manifest()
        self.features_path = manifest["features_path"]
        self.checkpoint_path = manifest["checkpoint_path"]
        self.hparams_path = manifest.get("hparams_path")

        print(f"\n📄 Using manifest:")
        print(f"   features_path   : {self.features_path}")
        print(f"   checkpoint_path : {self.checkpoint_path}")
        if self.hparams_path and os.path.exists(self.hparams_path):
            print(f"   hparams_path    : {self.hparams_path}")

        # 2) Dataset → graph
        print(f"\n📊 Loading stock data from: {data_path}")
        self.dataset = FinancialDataset(
            csv_folder_path=data_path,
            max_stocks=550
        )
        full_graph, self.metadata = self.dataset.prepare_dataset()

        # 3) Apply feature mask
        self.selected_idx = load_feature_indices(self.features_path)
        masked_graph = apply_feature_mask_to_graph(full_graph, self.selected_idx)
        self.graph_data = masked_graph

        # Update metadata with masked dimension
        self.metadata = dict(self.metadata)
        self.metadata['num_features'] = int(len(self.selected_idx))

        print(f"\n✅ Data ready:")
        print(f"   Masked features: {self.metadata['num_features']}")
        print(f"   Stocks: {self.metadata['num_stocks']}")
        print(f"   Edges: {self.metadata['num_edges']}")

        self.model = self._load_model()

        print("\n✅ Predictor ready!")
        print("="*80)

    def _load_model(self):
        """Load trained model matching the masked input dimension"""
        print(f"\n🤖 Loading model from: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Derive hparams if provided
        hidden_dim = 128
        dropout = 0.3
        if self.hparams_path and os.path.exists(self.hparams_path):
            try:
                with open(self.hparams_path, "r") as f:
                    h = json.load(f)
                hidden_dim = int(h.get("hidden_dim", hidden_dim))
                dropout = float(h.get("dropout", dropout))
            except Exception:
                pass

        input_dim = self.metadata['num_features']
        model = ImprovedFinGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )

        # Handle Lightning-style state dicts
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()

        print(f"✅ Model loaded with input_dim={input_dim}, hidden_dim={hidden_dim}, dropout={dropout}")
        return model

    @torch.no_grad()
    def predict_all_stocks(self):
        """Generate predictions for all stocks"""
        print("\n" + "="*80)
        print(f"📈 GENERATING PREDICTIONS FOR TOMORROW")
        print(f"   Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
        print("="*80)

        print("\n🔮 Running model inference...")

        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)

        returns_pred, movement_logits, ranking_scores, _ = self.model(x, edge_index)

        movement_probs = torch.softmax(movement_logits, dim=1)
        prob_up = movement_probs[:, 1].cpu().numpy()

        # Confidence is the distance from 0.5 scaled as a percentage
        confidence_pct = np.abs(prob_up - 0.5) * 200.0
        pred_dir = (prob_up >= 0.5).astype(int)

        print(f"\n📊 UP probability range: {prob_up.min():.3f} - {prob_up.max():.3f}")

        returns_pred = returns_pred.cpu().numpy().flatten()
        ranking_scores = ranking_scores.cpu().numpy().flatten()

        tickers = list(self.metadata['idx_to_ticker'].values())
        sectors = [self.metadata['sectors'].get(t, 'Other') for t in tickers]

        results_df = pd.DataFrame({
            'Ticker': tickers[:len(prob_up)],
            'Sector': sectors[:len(prob_up)],
            'Direction': ['UP' if d == 1 else 'DOWN' for d in pred_dir],
            'Confidence_%': confidence_pct[:len(prob_up)],
            'Expected_Return_%': (returns_pred[:len(prob_up)] * 100.0),
            'Ranking_Score': ranking_scores[:len(prob_up)],
            'UP_Probability': prob_up[:len(prob_up)],
        })

        # UP-only with probability threshold
        up_only = results_df[(results_df['Direction'] == 'UP') & (results_df['UP_Probability'] >= CONF_MIN)].copy()

        # Sort by ranking score
        up_only = up_only.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
        up_only['Rank'] = range(1, len(up_only) + 1)

        total = len(results_df)
        up_count = (results_df['Direction'] == 'UP').sum()
        down_count = total - up_count

        print(f"\n✅ Prediction Balance (before filtering):")
        print(f"   UP: {up_count} ({up_count/total*100:.1f}%)")
        print(f"   DOWN: {down_count} ({down_count/total*100:.1f}%)")
        print(f"   After UP+confidence filter: {len(up_only)} picks (Prob_UP ≥ {CONF_MIN:.2f})")

        return results_df, up_only

    def save_outputs(self, full_df, up_df):
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save full and filtered
        full_path = os.path.join(OUTPUT_DIR, f"predictions_{ts}.csv")
        up_path = os.path.join(OUTPUT_DIR, f"predictions_up_only_{ts}.csv")
        full_df.to_csv(full_path, index=False)
        up_df.to_csv(up_path, index=False)

        # Save Top-K
        paths = {}
        for k in TOPK_LIST:
            topk = up_df.head(k)
            p = os.path.join(OUTPUT_DIR, f"top{k}_{ts}.csv")
            topk.to_csv(p, index=False)
            paths[f"top{k}"] = p

        print(f"\n💾 Saved:")
        print(f"   Full: {full_path}")
        print(f"   UP-only: {up_path}")
        for k, p in paths.items():
            print(f"   {k.upper()}: {p}")

        return full_path, up_path, paths

    def print_topk(self, up_df):
        for k in TOPK_LIST:
            topk = up_df.head(k)
            print(f"\n=== TOP-{k} (UP-only, Prob_UP ≥ {CONF_MIN:.2f}) ===")
            if topk.empty:
                print("No picks passed the filter.")
                continue
            cols = ["Rank", "Ticker", "Sector", "UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]
            view = topk[cols].copy()
            for c in ["UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]:
                view[c] = view[c].map(lambda x: f"{x:.3f}")
            print(view.to_string(index=False))


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────

def main():
    try:
        data_path = 'indian_data'
        predictor = TomorrowPredictor(data_path)

        full_df, up_df = predictor.predict_all_stocks()

        # Optional: sector diversification on UP picks
        # up_df = diversify_by_sector(up_df, top_k=20, max_per_sector=3)

        predictor.print_topk(up_df)
        predictor.save_outputs(full_df, up_df)

        print("\n" + "="*80)
        print("✅ PREDICTION COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TIP: Ensure latest_manifest.json exists and paths are valid.")


if __name__ == "__main__":
    main()