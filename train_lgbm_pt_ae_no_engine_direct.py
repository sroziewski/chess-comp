import pandas as pd
import numpy as np
import chess
from chevy.features import KingSafety, PawnStructure, BoardFeatures  # Using Chevy again
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import gc
import warnings
import os
import shutil

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- Progress Bar Import ---
from tqdm.auto import tqdm

# --- Configuration Import ---
from chess_puzzle_rating.utils.config import get_config

tqdm.pandas()

warnings.filterwarnings('ignore', category=UserWarning, module='chevy')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Load Configuration ---
config = get_config()

# --- Data Paths ---
TRAIN_FILE = config['data_paths']['train_file']
TEST_FILE = config['data_paths']['test_file']
SUBMISSION_FILE = config['data_paths']['submission_file']

# --- Training Parameters ---
N_SPLITS_LGBM = config['training']['n_splits_lgbm']
RANDOM_STATE = config['training']['random_state']
LGBM_EARLY_STOPPING_ROUNDS = config['training']['lgbm_early_stopping_rounds']

# --- PyTorch Autoencoder Training Config ---
AE_EPOCHS = config['autoencoder']['epochs']
AE_BATCH_SIZE_PER_GPU = config['autoencoder']['batch_size_per_gpu']
AE_LEARNING_RATE = config['autoencoder']['learning_rate']
AE_VALID_SPLIT = config['autoencoder']['valid_split']
AE_EARLY_STOPPING_PATIENCE = config['autoencoder']['early_stopping_patience']
MODEL_SAVE_DIR = config['autoencoder']['model_save_dir']
AE_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "autoencoders")
LGBM_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "lightgbm_folds")
FORCE_RETRAIN_AES = config['autoencoder']['force_retrain']

os.makedirs(AE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LGBM_MODEL_SAVE_DIR, exist_ok=True)

# --- GPU Configuration ---
TARGET_NUM_GPUS = config['gpu']['target_num_gpus']
if torch.cuda.is_available():
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Found {AVAILABLE_GPUS} CUDA GPUs.")
    GPUS_TO_USE = min(AVAILABLE_GPUS, TARGET_NUM_GPUS)
    if GPUS_TO_USE > 0:
        DEVICE_IDS = list(range(GPUS_TO_USE))
        PRIMARY_DEVICE_STR = f'cuda:{DEVICE_IDS[0]}'
        print(f"Will use {GPUS_TO_USE} GPUs: {DEVICE_IDS} for PyTorch. Primary: {PRIMARY_DEVICE_STR}")
        AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU * GPUS_TO_USE
        print(f"PyTorch Autoencoder Total Batch Size for Training: {AE_TOTAL_BATCH_SIZE} (across {GPUS_TO_USE} GPUs)")
    else:
        DEVICE_IDS = None
        PRIMARY_DEVICE_STR = 'cpu'
        print("CUDA available but 0 GPUs selected. Using CPU.")
        AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU
else:
    AVAILABLE_GPUS = 0
    GPUS_TO_USE = 0
    DEVICE_IDS = None
    PRIMARY_DEVICE_STR = 'cpu'
    print("No CUDA GPUs. Using CPU for PyTorch.")
    AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU
DEVICE = torch.device(PRIMARY_DEVICE_STR)

# --- PyTorch Autoencoder Definition ---
PROB_SEQ_LENGTH = config['autoencoder']['sequence_length']
EMBEDDING_DIM_PROB = config['autoencoder']['embedding_dim']


class ProbAutoencoder(nn.Module):
    def __init__(self, input_seq_len=PROB_SEQ_LENGTH, embedding_dim=EMBEDDING_DIM_PROB):
        super(ProbAutoencoder, self).__init__()
        # Encoder layers
        self.encoder_conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.encoder_bn1 = nn.BatchNorm1d(32)
        self.encoder_conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.encoder_bn2 = nn.BatchNorm1d(64)
        self.encoder_conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.encoder_bn3 = nn.BatchNorm1d(64)

        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)

        self.encoder_pool = nn.AdaptiveMaxPool1d(1)

        # VAE: separate layers for mean and log variance
        self.encoder_fc_mu = nn.Linear(64, embedding_dim)
        self.encoder_fc_logvar = nn.Linear(64, embedding_dim)

        # Decoder layers
        self.decoder_fc_expand = nn.Linear(embedding_dim, 128)
        self.decoder_relu = nn.ReLU()
        self.decoder_fc_mid = nn.Linear(128, 64)
        self.decoder_fc_final = nn.Linear(64, input_seq_len)

        # Store embedding dimension for KL divergence calculation
        self.embedding_dim = embedding_dim

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = F.relu(self.encoder_bn3(self.encoder_conv3(x)))

        # Apply attention mechanism
        # Reshape for attention: [batch, channels, seq_len] -> [batch, seq_len, channels]
        x_reshaped = x.permute(0, 2, 1)
        x_attn, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        # Reshape back: [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x_attn.permute(0, 2, 1)

        x = self.encoder_pool(x)
        x = x.view(x.size(0), -1)

        # Return mean and log variance
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)

        return mu, logvar

    def decode(self, z):
        x = self.decoder_relu(self.decoder_fc_expand(z))
        x = self.decoder_relu(self.decoder_fc_mid(x))
        return torch.sigmoid(self.decoder_fc_final(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # Store KL divergence for loss calculation
        # KL divergence between N(mu, var) and N(0, 1)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return recon_x, z, mu, logvar, kl_div


# --- Contrastive Loss Function ---
def contrastive_loss(embeddings, temperature=0.5):
    """
    Implements contrastive loss (InfoNCE) for better embeddings

    Args:
        embeddings: Batch of embeddings [batch_size, embedding_dim]
        temperature: Temperature parameter for scaling

    Returns:
        Contrastive loss value
    """
    batch_size = embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)  # Can't compute contrastive loss with single sample

    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / temperature

    # Mask out self-similarity
    mask = torch.eye(batch_size, device=similarity_matrix.device)
    mask = 1 - mask  # Invert to keep non-diagonal elements

    # For numerical stability, subtract max from each row
    similarity_matrix = similarity_matrix * mask
    similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()

    # Compute positive and negative pairs
    exp_sim = torch.exp(similarity_matrix) * mask

    # Sum over all negatives
    sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)

    # Compute log-softmax
    log_prob = -torch.log(exp_sim / (sum_exp_sim + 1e-8) + 1e-8)

    # Average over all positive pairs
    loss = log_prob.sum() / (batch_size * (batch_size - 1))

    return loss

# --- PyTorch Autoencoder Training Function ---
def train_autoencoder(auto_model_base, train_loader, val_loader, epochs, lr, patience, model_save_path,
                      model_name="Autoencoder"):
    if GPUS_TO_USE > 1 and DEVICE_IDS:
        print(f"Wrapping {model_name} with DataParallel for training on GPUs: {DEVICE_IDS}")
        auto_model_train = nn.DataParallel(auto_model_base, device_ids=DEVICE_IDS)
    else:
        auto_model_train = auto_model_base

    # Reconstruction loss
    recon_criterion = nn.MSELoss(reduction='sum')

    # Loss weights from configuration
    kl_weight = config['autoencoder']['kl_weight']  # Weight for KL divergence loss
    contrastive_weight = config['autoencoder']['contrastive_weight']  # Weight for contrastive loss

    optimizer = optim.Adam(auto_model_train.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    print(
        f"\n--- Training {model_name} on {DEVICE if not isinstance(auto_model_train, nn.DataParallel) else 'multiple GPUs'} ---")
    print(f"Model will be saved to: {model_save_path}")

    for epoch in range(epochs):
        auto_model_train.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_contrastive_loss = 0.0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [T]", leave=False, unit="batch")

        for data_batch in progress_bar_train:
            inputs = data_batch[0].to(DEVICE)
            optimizer.zero_grad()

            # Forward pass with VAE
            recon_x, z, mu, logvar, kl_div = auto_model_train(inputs)

            # Reconstruction loss
            recon_loss = recon_criterion(recon_x, inputs) / inputs.size(0)

            # KL divergence loss
            kl_loss = kl_div.mean()

            # Contrastive loss
            contr_loss = contrastive_loss(z)

            # Total loss
            loss = recon_loss + kl_weight * kl_loss + contrastive_weight * contr_loss

            loss.backward()
            optimizer.step()

            # Track losses
            train_loss += loss.item() * inputs.size(0)
            train_recon_loss += recon_loss.item() * inputs.size(0)
            train_kl_loss += kl_loss.item() * inputs.size(0)
            train_contrastive_loss += contr_loss.item() * inputs.size(0)

            progress_bar_train.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'contr': f'{contr_loss.item():.4f}'
            })

        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_kl_loss /= len(train_loader.dataset)
        train_contrastive_loss /= len(train_loader.dataset)

        auto_model_train.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_contrastive_loss = 0.0

        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [V]", leave=False, unit="batch")

        with torch.no_grad():
            for data_batch in progress_bar_val:
                inputs = data_batch[0].to(DEVICE)

                # Forward pass with VAE
                recon_x, z, mu, logvar, kl_div = auto_model_train(inputs)

                # Reconstruction loss
                recon_loss = recon_criterion(recon_x, inputs) / inputs.size(0)

                # KL divergence loss
                kl_loss = kl_div.mean()

                # Contrastive loss
                contr_loss = contrastive_loss(z)

                # Total loss
                loss = recon_loss + kl_weight * kl_loss + contrastive_weight * contr_loss

                # Track losses
                val_loss += loss.item() * inputs.size(0)
                val_recon_loss += recon_loss.item() * inputs.size(0)
                val_kl_loss += kl_loss.item() * inputs.size(0)
                val_contrastive_loss += contr_loss.item() * inputs.size(0)

                progress_bar_val.set_postfix({
                    'val_loss': f'{loss.item():.4f}'
                })

        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)
        val_kl_loss /= len(val_loader.dataset)
        val_contrastive_loss /= len(val_loader.dataset)

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}, Contr: {train_contrastive_loss:.6f}), "
            f"Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f}, Contr: {val_contrastive_loss:.6f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save_state = auto_model_train.module.state_dict() if isinstance(auto_model_train,
                                                                                     nn.DataParallel) else auto_model_train.state_dict()
            torch.save(model_to_save_state, model_save_path)
            tqdm.write(f"Best AE model saved: {model_save_path} (Val Loss: {best_val_loss:.6f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            tqdm.write(f"Val loss not improved for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            tqdm.write(f"AE Early stopping: {epoch + 1} epochs.")
            break

    print(f"--- Training for {model_name} finished. ---")
    print(f"Loading best AE model from {model_save_path} (Best Val Loss Recorded: {best_val_loss:.6f})")
    auto_model_base.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    auto_model_base.eval()
    return auto_model_base


# --- Robust Feature Engineering Functions (using Chevy) ---
def get_chevy_features_for_color(board, color_to_eval, prefix):
    features = {}
    try:
        ks = KingSafety(board, color=color_to_eval)
        features[prefix + 'king_mobility'] = ks.king_mobility
        features[prefix + 'castling_rights_bool'] = int(ks.castling_rights)
        features[prefix + 'king_centrality'] = ks.king_centrality
        features[prefix + 'checked'] = int(ks.checked)
        features[prefix + 'king_attackers_ring_1'] = ks.king_attackers_looking_at_ring_1
        features[prefix + 'king_defenders_ring_1'] = ks.king_defenders_at_ring_1
    except Exception as e_ks:
        # print(f"DEBUG: Chevy KingSafety error for FEN {board.fen()}, Color {color_to_eval}, Prefix {prefix}: {e_ks}")
        pass  # Continue if a specific Chevy part fails

    try:
        ps = PawnStructure(board, color=color_to_eval)
        features[prefix + 'passed_pawns'] = ps.passed_pawns
        features[prefix + 'isolated_pawns'] = ps.isolated_pawns
        features[prefix + 'doubled_pawns'] = ps.doubled_pawns
        features[prefix + 'blocked_pawns'] = ps.blocked_pawns
    except Exception as e_ps:
        # print(f"DEBUG: Chevy PawnStructure error for FEN {board.fen()}, Color {color_to_eval}, Prefix {prefix}: {e_ps}")
        pass

    try:
        bf = BoardFeatures(board, color=color_to_eval)
        features[prefix + 'bishop_pair'] = int(bf.bishop_pair)
        features[prefix + 'queens_mobility'] = getattr(bf, 'queens_mobility', 0)
        features[prefix + 'rooks_mobility'] = getattr(bf, 'rooks_mobility', 0)
        features[prefix + 'knights_mobility'] = getattr(bf, 'knights_mobility', 0)
        features[prefix + 'bishops_mobility'] = getattr(bf, 'bishops_mobility', 0)
        features[prefix + 'connectivity'] = bf.connectivity
    except Exception as e_bf:
        # print(f"DEBUG: Chevy BoardFeatures error for FEN {board.fen()}, Color {color_to_eval}, Prefix {prefix}: {e_bf}")
        pass

    return features  # Always return a dictionary


def get_material_and_piece_counts(board, color_to_eval, prefix):
    features = {}
    material = 0
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    try:
        for piece_type in chess.PIECE_TYPES:
            count = len(board.pieces(piece_type, color_to_eval))
            features[prefix + f'{chess.piece_name(piece_type)}_count'] = count
            if piece_type in piece_values:
                material += count * piece_values[piece_type]
        features[prefix + 'material'] = material
    except Exception as e_mat:
        # print(f"DEBUG: Material count error for FEN {board.fen()}, Color {color_to_eval}, Prefix {prefix}: {e_mat}")
        pass  # Default material to 0 and counts to potentially 0 or missing

    return features


def get_extended_fen_features(fen_string):
    all_features = {}
    try:
        board = chess.Board(fen_string)
    except ValueError:
        # print(f"DEBUG: Invalid FEN skipped: {fen_string}")
        return {}  # Return empty dict for invalid FENs, FE will result in NaNs for these rows

    active_color, opponent_color = board.turn, not board.turn

    active_chevy_data = get_chevy_features_for_color(board, active_color, 'active_')
    all_features.update(active_chevy_data)  # Safe as get_chevy_features_for_color always returns dict

    opponent_chevy_data = get_chevy_features_for_color(board, opponent_color, 'opponent_')
    all_features.update(opponent_chevy_data)

    active_material_data = get_material_and_piece_counts(board, active_color, 'active_')
    all_features.update(active_material_data)

    opponent_material_data = get_material_and_piece_counts(board, opponent_color, 'opponent_')
    all_features.update(opponent_material_data)

    all_features['material_diff'] = all_features.get('active_material', 0) - all_features.get('opponent_material', 0)
    all_features['total_material'] = all_features.get('active_material', 0) + all_features.get('opponent_material', 0)

    try:
        all_features['active_player_legal_moves'] = len(list(board.legal_moves))
    except Exception:
        all_features['active_player_legal_moves'] = 0

    all_features['can_castle_WK'] = int(board.has_kingside_castling_rights(chess.WHITE))
    all_features['can_castle_WQ'] = int(board.has_queenside_castling_rights(chess.WHITE))
    all_features['can_castle_BK'] = int(board.has_kingside_castling_rights(chess.BLACK))
    all_features['can_castle_BQ'] = int(board.has_queenside_castling_rights(chess.BLACK))
    all_features['active_player_is_white'] = int(active_color == chess.WHITE)
    all_features['halfmove_clock'] = board.halfmove_clock
    all_features['fullmove_number'] = board.fullmove_number

    return all_features


def get_moves_features(moves_string):
    if pd.isna(moves_string) or not moves_string:
        return {
            'solution_num_half_moves': 0,
            'solution_num_captures': 0,
            'solution_num_checks': 0,
            'solution_is_checkmate': 0
        }

    solution_half_moves = moves_string.split(' ')
    features = {'solution_num_half_moves': len(solution_half_moves)}
    features['solution_num_captures'] = sum(1 for m in solution_half_moves if 'x' in m)
    features['solution_num_checks'] = sum(1 for m in solution_half_moves if '+' in m)
    features['solution_is_checkmate'] = int('#' in solution_half_moves[-1]) if solution_half_moves else 0

    return features


def extract_embeddings_from_autoencoder(autoencoder_model, sequences_np, current_device,
                                        batch_size_per_gpu=AE_BATCH_SIZE_PER_GPU,
                                        num_gpus_for_inference=GPUS_TO_USE):
    if autoencoder_model is None:  # Handle case where AE was not trained/loaded
        print("Warning: Autoencoder model is None in extract_embeddings_from_autoencoder. Returning empty array.")
        return np.array([])

    if sequences_np.shape[0] == 0:
        # For VAE, we use encoder_fc_mu instead of encoder_fc
        out_dim = autoencoder_model.module.encoder_fc_mu.out_features if isinstance(autoencoder_model,
                                                                                 nn.DataParallel) else autoencoder_model.encoder_fc_mu.out_features
        return np.array([]).reshape(0, out_dim)

    model_for_inference = autoencoder_model  # This should be the base model on DEVICE
    # If multi-GPU inference is desired and model isn't already DataParallel (it shouldn't be if train_autoencoder returns base)
    if num_gpus_for_inference > 1 and DEVICE_IDS and not isinstance(autoencoder_model, nn.DataParallel):
        print(f"Wrapping AE model with DataParallel for inference on GPUs: {DEVICE_IDS}")
        model_for_inference = nn.DataParallel(autoencoder_model, device_ids=DEVICE_IDS)

    actual_inference_batch_size = batch_size_per_gpu * num_gpus_for_inference if isinstance(model_for_inference,
                                                                                            nn.DataParallel) else batch_size_per_gpu

    dataset = TensorDataset(torch.tensor(sequences_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=actual_inference_batch_size, shuffle=False, num_workers=2,
                        pin_memory=(current_device.type == 'cuda'))
    all_embeddings_list = []
    model_for_inference.eval()

    with torch.no_grad():
        for batch_data_tuple in tqdm(loader, desc="Extracting Embeddings", leave=False, unit="batch"):
            inputs = batch_data_tuple[0].to(current_device)
            # VAE returns (recon_x, z, mu, logvar, kl_div)
            # We use z (the sampled latent vector) as our embedding
            outputs = model_for_inference(inputs)

            # Extract the latent representation (z) which is the second element in the output tuple
            embeddings = outputs[1]  # z is at index 1
            all_embeddings_list.append(embeddings.cpu().numpy())

    if not all_embeddings_list:
        # For VAE, we use encoder_fc_mu instead of encoder_fc
        out_dim = autoencoder_model.encoder_fc_mu.out_features  # Access from base model
        return np.array([]).reshape(0, out_dim)

    return np.concatenate(all_embeddings_list, axis=0)


def get_success_prob_features_with_trained_ae(df_in, trained_rapid_ae, trained_blitz_ae, device_for_loading):
    df = df_in.copy()
    prob_cols = [col for col in df.columns if 'success_prob_' in col]
    rapid_prob_cols = sorted([col for col in prob_cols if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    blitz_prob_cols = sorted([col for col in prob_cols if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))
    features_df = pd.DataFrame(index=df.index)

    if not prob_cols:
        return features_df

    features_df['prob_all_mean'] = df[prob_cols].mean(axis=1)
    features_df['prob_all_std'] = df[prob_cols].std(axis=1)
    features_df['prob_rapid_mean'] = df[rapid_prob_cols].mean(axis=1) if rapid_prob_cols else np.nan
    features_df['prob_blitz_mean'] = df[blitz_prob_cols].mean(axis=1) if blitz_prob_cols else np.nan

    if trained_rapid_ae and rapid_prob_cols and len(rapid_prob_cols) == PROB_SEQ_LENGTH:
        rapid_sequences = df[rapid_prob_cols].dropna().values.astype(np.float32)  # dropna here too
        if rapid_sequences.shape[0] > 0:
            print("Extracting Rapid Prob Embeddings (trained AE)...")
            rapid_embeddings_np = extract_embeddings_from_autoencoder(trained_rapid_ae, rapid_sequences,
                                                                      device_for_loading)
            if rapid_embeddings_np.size > 0:  # Check if embeddings were generated
                for i in range(rapid_embeddings_np.shape[1]):
                    features_df[f'pt_ae_rapid_emb_{i}'] = rapid_embeddings_np[:, i]
            else:
                print("No rapid embeddings generated (empty result).")

    if trained_blitz_ae and blitz_prob_cols and len(blitz_prob_cols) == PROB_SEQ_LENGTH:
        blitz_sequences = df[blitz_prob_cols].dropna().values.astype(np.float32)  # dropna here too
        if blitz_sequences.shape[0] > 0:
            print("Extracting Blitz Prob Embeddings (trained AE)...")
            blitz_embeddings_np = extract_embeddings_from_autoencoder(trained_blitz_ae, blitz_sequences,
                                                                      device_for_loading)
            if blitz_embeddings_np.size > 0:  # Check if embeddings were generated
                for i in range(blitz_embeddings_np.shape[1]):
                    features_df[f'pt_ae_blitz_emb_{i}'] = blitz_embeddings_np[:, i]
            else:
                print("No blitz embeddings generated (empty result).")

    return features_df


def process_text_tags(df_series, prefix, min_df=5):
    print(f"Vectorizing {prefix}...")
    series_str = df_series.astype(str).fillna(f'Unknown{prefix.capitalize()}')
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '), binary=True, min_df=min_df)
    encoded_matrix = vectorizer.fit_transform(series_str)
    feature_names = [f'{prefix}_{name.replace(" ", "_")}' for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(encoded_matrix.toarray(), columns=feature_names, index=df_series.index)


# --- Main Script ---
if __name__ == '__main__':
    print(f"Using PyTorch on device: {DEVICE} with {GPUS_TO_USE} GPU(s) for DataParallel if > 1.")
    print("Loading data...")
    train_df_orig = pd.read_csv(TRAIN_FILE)
    test_df_orig = pd.read_csv(TEST_FILE)
    test_puzzle_ids = test_df_orig['PuzzleId']
    train_df_orig['is_train'] = 1
    test_df_orig['is_train'] = 0

    if 'Rating' not in test_df_orig.columns:
        test_df_orig['Rating'] = np.nan

    combined_df = pd.concat([train_df_orig, test_df_orig], ignore_index=True, sort=False)

    prob_cols_all = [col for col in combined_df.columns if 'success_prob_' in col]
    rapid_prob_cols_all = sorted([col for col in prob_cols_all if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    blitz_prob_cols_all = sorted([col for col in prob_cols_all if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))

    trained_rapid_ae, trained_blitz_ae = None, None
    rapid_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "rapid_ae_best.pth")
    blitz_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "blitz_ae_best.pth")

    for ae_type, prob_cols_subset, save_path, model_var_name in [
        ("RapidProbAE", rapid_prob_cols_all, rapid_ae_save_path, "trained_rapid_ae"),
        ("BlitzProbAE", blitz_prob_cols_all, blitz_ae_save_path, "trained_blitz_ae")
    ]:
        current_ae_model = None
        if os.path.exists(save_path) and not FORCE_RETRAIN_AES:
            print(f"Loading pre-trained {ae_type} model from {save_path}...")
            current_ae_model = ProbAutoencoder().to(DEVICE)
            current_ae_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            current_ae_model.eval()
            print(f"{ae_type} model loaded successfully.")
        elif prob_cols_subset and len(prob_cols_subset) == PROB_SEQ_LENGTH:
            # Ensure we use .dropna() on the correct subset of columns *before* .values
            seq_data_all_for_current_ae = combined_df[prob_cols_subset].dropna().values.astype(np.float32)
            if seq_data_all_for_current_ae.shape[0] > AE_TOTAL_BATCH_SIZE:
                X_tr, X_val = train_test_split(seq_data_all_for_current_ae, test_size=AE_VALID_SPLIT,
                                               random_state=RANDOM_STATE)
                loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=AE_TOTAL_BATCH_SIZE,
                                       shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
                loader_val = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=AE_TOTAL_BATCH_SIZE,
                                        shuffle=False, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
                base_ae_model = ProbAutoencoder().to(DEVICE)
                current_ae_model = train_autoencoder(base_ae_model, loader_tr, loader_val, AE_EPOCHS, AE_LEARNING_RATE,
                                                     AE_EARLY_STOPPING_PATIENCE, save_path, ae_type)
            else:
                print(
                    f"Not enough {ae_type.split('ProbAE')[0]} data for AE training (found {seq_data_all_for_current_ae.shape[0]} sequences).")
        else:
            print(f"{ae_type.split('ProbAE')[0]} prob columns not found/wrong length for AE.")

        if model_var_name == "trained_rapid_ae":
            trained_rapid_ae = current_ae_model
        elif model_var_name == "trained_blitz_ae":
            trained_blitz_ae = current_ae_model

    print("\nEngineering other features (FEN, Moves, Tags)...")
    fen_features_df = pd.DataFrame(combined_df['FEN'].progress_apply(get_extended_fen_features).tolist(),
                                   index=combined_df.index)
    combined_df = pd.concat([combined_df, fen_features_df], axis=1)
    del fen_features_df
    gc.collect()

    moves_features_df = pd.DataFrame(combined_df['Moves'].progress_apply(get_moves_features).tolist(),
                                     index=combined_df.index)
    combined_df = pd.concat([combined_df, moves_features_df], axis=1)
    del moves_features_df
    gc.collect()

    # Use configuration values for text vectorization
    themes_min_df = config['feature_engineering']['text_vectorization']['themes_min_df']
    openings_min_df = config['feature_engineering']['text_vectorization']['openings_min_df']

    themes_df = process_text_tags(combined_df['Themes'], prefix='theme', min_df=themes_min_df)
    combined_df = pd.concat([combined_df, themes_df], axis=1)
    del themes_df
    gc.collect()

    openings_df = process_text_tags(combined_df['OpeningTags'], prefix='opening', min_df=openings_min_df)
    combined_df = pd.concat([combined_df, openings_df], axis=1)
    del openings_df
    gc.collect()

    if trained_rapid_ae or trained_blitz_ae:
        # Pass only the relevant success_prob columns to the feature extractor
        success_prob_df_subset = combined_df[
            prob_cols_all] if prob_cols_all else pd.DataFrame()  # Handle if prob_cols_all is empty
        if not success_prob_df_subset.empty:
            success_prob_features_df = get_success_prob_features_with_trained_ae(
                success_prob_df_subset, trained_rapid_ae, trained_blitz_ae, DEVICE
            )
            combined_df = pd.concat([combined_df, success_prob_features_df], axis=1)
            del success_prob_features_df
            gc.collect()
        else:
            print("Warning: No success_prob columns found to pass to AE feature extractor.")
    else:
        print("Warning: No AEs available. Using only basic aggregates for success_prob.")
        if 'prob_all_mean' not in combined_df.columns and prob_cols_all:
            combined_df['prob_all_mean'] = combined_df[prob_cols_all].mean(axis=1)
            combined_df['prob_all_std'] = combined_df[prob_cols_all].std(axis=1)

    for col in ['Popularity', 'NbPlays']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
            min_val = combined_df.loc[combined_df['is_train'] == 1, col].min()
            if min_val <= 0:
                combined_df[f'{col}_log'] = np.log1p(combined_df[col] - (min_val if min_val < 0 else 0))
            else:
                combined_df[f'{col}_log'] = np.log(combined_df[col])

    print("\nPreparing data for LightGBM model...")
    target_col = 'Rating'
    original_cols_to_drop = ['PuzzleId', 'FEN', 'Moves', 'Themes', 'GameUrl', 'OpeningTags']

    if 'Popularity_log' in combined_df.columns:
        original_cols_to_drop.append('Popularity')
    if 'NbPlays_log' in combined_df.columns:
        original_cols_to_drop.append('NbPlays')

    feature_columns = [col for col in combined_df.columns if
                       col not in [target_col, 'is_train'] + original_cols_to_drop]
    numeric_feature_columns = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            numeric_feature_columns.append(col)
        else:
            print(f"Warning: Dropping non-numeric pre-LGBM: {col} (dtype: {combined_df[col].dtype})")

    feature_columns = numeric_feature_columns
    train_processed_df = combined_df[combined_df['is_train'] == 1].copy()
    test_processed_df = combined_df[combined_df['is_train'] == 0].copy()
    X_train = train_processed_df[feature_columns].astype(np.float32)
    y_train = train_processed_df[target_col].astype(np.float32)
    X_test = test_processed_df[feature_columns].astype(np.float32)
    X_train = X_train.fillna(-999.0)
    X_test = X_test.fillna(-999.0)
    print(f"Training LightGBM with {X_train.shape[1]} features.")

    # Create an explicit train/validation split for final validation metrics
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=N_SPLITS_LGBM, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(X_train.shape[0])
    test_preds_lgbm = np.zeros(X_test.shape[0])

    # Load LightGBM parameters from configuration
    lgb_params = config['lgbm_params'].copy()

    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kf.split(X_train, y_train), total=N_SPLITS_LGBM, desc="LGBM KFold Training")):
        lgb_params['seed'] = RANDOM_STATE + fold
        X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
        )

        model_fold_path = os.path.join(LGBM_MODEL_SAVE_DIR,
                                       f"lgbm_fold_{fold + 1}_best_iter_{model.best_iteration_}.txt")
        model.booster_.save_model(model_fold_path)
        tqdm.write(f"Fold {fold + 1} LGBM model saved: {model_fold_path} (Best iter: {model.best_iteration_})")

        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration_)
        test_preds_lgbm += model.predict(X_test, num_iteration=model.best_iteration_) / N_SPLITS_LGBM

    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nOverall OOF RMSE (LGBM with PyTorch AE, no engine): {oof_rmse:.4f}")

    # Train a model on the explicit train/validation split for validation metrics
    print("\nTraining model on explicit validation split for final metrics...")
    val_model = lgb.LGBMRegressor(**lgb_params)
    val_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
    )

    val_preds = val_model.predict(X_val_final, num_iteration=val_model.best_iteration_)
    val_rmse = np.sqrt(mean_squared_error(y_val_final, val_preds))
    print(f"Validation RMSE on explicit 80/20 split: {val_rmse:.4f}")

    # Additional detailed validation metrics
    val_mae = mean_absolute_error(y_val_final, val_preds)
    val_r2 = r2_score(y_val_final, val_preds)
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation R² Score: {val_r2:.4f}")

    # Calculate RMSE by rating range
    print("\nRMSE by rating range:")
    rating_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, float('inf'))]
    for low, high in rating_ranges:
        mask = (y_val_final >= low) & (y_val_final < high)
        if np.sum(mask) > 0:
            range_rmse = np.sqrt(mean_squared_error(y_val_final[mask], val_preds[mask]))
            print(f"  {low}-{high}: {range_rmse:.4f} (n={np.sum(mask)})")

    print("Generating submission file...")
    final_predictions = np.round(test_preds_lgbm).astype(int)
    submission_df = pd.DataFrame({'PuzzleId': test_puzzle_ids, 'Rating': final_predictions})
    submission_file_path = SUBMISSION_FILE

    if not submission_file_path.lower().endswith('.txt'):
        submission_file_path = os.path.splitext(submission_file_path)[0] + ".txt"

    with open(submission_file_path, 'w') as f:
        for pred_rating in submission_df['Rating']:
            f.write(f"{pred_rating}\n")

    print(f"Submission file '{submission_file_path}' created.")
    print("First 5 predictions:\n", submission_df.head())
