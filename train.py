import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import from your other files
from models import SummDPPLSTM
from losses import DPPLoss
from dataset import get_dataloaders

def train_two_phase(train_loader, nx=1024, nh=256, nout=256, device='cuda',
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
    
    for epoch in range(50):
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
            
        if (epoch + 1) % 10 == 0:
            print(f"Phase 1 - Epoch {epoch+1}, MSE Loss: {total_loss / len(train_loader):.4f}")

    # Save Phase 1 weights        
    Path(phase1_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), phase1_save_path)

    # ==========================================
    # PHASE 2: Train dppLSTM (Diversity)
    # ==========================================
    print("\n--- Starting Phase 2: Training dppLSTM (Diversity) ---")
    # Load the converged Phase 1 weights
    model.load_state_dict(torch.load(phase1_save_path))
    
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5)
    dpp_weight = 0.1 
    
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        
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
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Phase 2 - Epoch {epoch+1}, Total Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), final_save_path)
    print("Training Complete. Final model saved.")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path to the Kaggle file (Change this to test TVSum, OVP, or YouTube!)
    kaggle_h5_path = '/kaggle/input/summe-video-summarization/eccv16_dataset_summe_google_pool5.h5'
    
    print(f"Loading data from {kaggle_h5_path}...")
    train_loader, test_loader = get_dataloaders(kaggle_h5_path, batch_size=1)
    
    train_two_phase(train_loader, device=device)