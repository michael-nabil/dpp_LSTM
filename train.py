import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import from your other files
from models import SummDPPLSTM
from losses import DPPLoss
from dataset import get_dataloaders

def train_two_phase(train_loader, val_loader, nx=1024, nh=256, nout=256, device='cuda',
                    phase1_save_path='./saved_models/vsLSTM_phase1.pt', 
                    final_save_path='./saved_models/dppLSTM_final.pt'):
    
    model = SummDPPLSTM(nx=nx, nh=nh, nout=nout).to(device)
    
    criterion_score = nn.MSELoss()
    criterion_dpp = DPPLoss()

    # ==========================================
    # PHASE 1: Train vsLSTM (Importance Only)
    # ==========================================
    print("--- Starting Phase 1: Training vsLSTM (Importance) ---")
    optimizer_phase1 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Early Stopping variables
    best_val_loss_p1 = float('inf')
    patience_counter_p1 = 0
    patience_limit_p1 = 15 # Stop if no improvement for 15 epochs
    max_epochs_p1 = 50

    for epoch in range(max_epochs_p1):
        # Training Loop
        model.train()
        total_loss = 0.0
        
        # PyTorch Iterator loop replaces the old Theano index loop
        for video_features, gt_score, _ in train_loader:
            # Squeeze removes the dummy batch dimension created by DataLoader
            video_features = video_features.squeeze(0).to(device)
            gt_score = gt_score.squeeze(0).to(device)
            
            optimizer_phase1.zero_grad()
            q_score, _ = model(video_features)
            
            # Phase 1 only cares about matching the ground truth importance scores
            loss = criterion_score(q_score.squeeze(), gt_score)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_phase1.step()
            total_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for video_features, gt_score, _ in val_loader:
                video_features = video_features.squeeze(0).to(device)
                gt_score = gt_score.squeeze(0).to(device)
                
                q_score, _ = model(video_features)
                
                # Calculate validation MSE
                loss = criterion_score(q_score.squeeze(), gt_score)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch metrics, and Checking for Early Stopping
        print(f"Phase 1 - Epoch [{epoch+1}/{max_epochs_p1}] | Train MSE(loss): {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss_p1:
            print(f"   >> Phase 1 Val loss improved from {best_val_loss_p1:.4f} to {avg_val_loss:.4f}. Saving vsLSTM!")
            best_val_loss_p1 = avg_val_loss
            patience_counter_p1 = 0
            
            # Save the best Phase 1 weights securely
            Path(phase1_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), phase1_save_path)
        else:
            patience_counter_p1 += 1
            print(f"   >> No improvement. Patience: {patience_counter_p1}/{patience_limit_p1}")
            
            if patience_counter_p1 >= patience_limit_p1:
                print(f"--- Phase 1 Early stopping triggered at epoch {epoch+1}! ---")
                break # Exit the Phase 1 training loop

        print(f"Phase 1 Complete. Best vsLSTM Validation MSE: {best_val_loss_p1:.4f}")  # Model is saved in the Early Stopping Checking

    # ==========================================
    # PHASE 2: Train dppLSTM (Diversity)
    # ==========================================
    print("\n--- Starting Phase 2: Training dppLSTM (Diversity) ---")
    # Load the converged Phase 1 weights
    model.load_state_dict(torch.load(phase1_save_path))
    
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5)
    dpp_weight = 0.1 
    
    # Variables for Monitoring each epoch's validation loss, for using Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15 # Stop if no improvement for 15 epochs
    max_epochs = 100

    for epoch in range(50):
        
        # Training Loop
        model.train()
        train_loss = 0.0
        
        for video_features, gt_score, gt_summary in train_loader:
            video_features = video_features.squeeze(0).to(device)
            gt_score = gt_score.squeeze(0).to(device)
            gt_summary = gt_summary.squeeze(0).to(device)
            
            optimizer_phase2.zero_grad()
            q_score, pred_k = model(video_features)
            
            loss_score = criterion_score(q_score.squeeze(), gt_score)
            loss_dpp = criterion_dpp(pred_k, gt_summary)
            
            loss = loss_score + (dpp_weight * loss_dpp)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_phase2.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for speed and memory
            for video_features, gt_score, gt_summary in val_loader:
                video_features = video_features.squeeze(0).to(device)
                gt_score = gt_score.squeeze(0).to(device)
                gt_summary = gt_summary.squeeze(0).to(device)
                
                q_score, pred_k = model(video_features)
                
                loss_score = criterion_score(q_score.squeeze(), gt_score)
                loss_dpp = criterion_dpp(pred_k, gt_summary)
                
                loss = loss_score + (dpp_weight * loss_dpp)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch metrics, and Checking for Early Stopping

        print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"   >> Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model!")
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the model safely
            Path(final_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), final_save_path)
        else:
            patience_counter += 1
            print(f"   >> No improvement. Patience: {patience_counter}/{patience_limit}")
            
            if patience_counter >= patience_limit:
                print(f"--- Early stopping triggered at epoch {epoch+1}! ---")
                break # Exit the training loop early

    print(f"Phase 2 Complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path to the Kaggle file (Change this to test TVSum, OVP, or YouTube!)
    kaggle_h5_path = '/kaggle/input/summe-video-summarization/eccv16_dataset_summe_google_pool5.h5'
    
    print(f"Loading data from {kaggle_h5_path}...")
    train_loader, test_loader = get_dataloaders(kaggle_h5_path, batch_size=1)
    
    train_two_phase(train_loader, device=device)