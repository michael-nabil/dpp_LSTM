import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Assume models.py and losses.py are in the same directory
from models import SummDPPLSTM
from losses import DPPLoss
# from tools.data_loader import load_data  <-- Ensure this provides [features, scores, summary]

def train_two_phase(train_set, val_set, nx=1024, nh=256, nout=256, device='cuda'):
    
    model = SummDPPLSTM(nx=nx, nh=nh, nout=nout).to(device)
    
    criterion_score = nn.MSELoss()
    criterion_dpp = DPPLoss()
    
    save_dir = Path('./saved_models')
    save_dir.mkdir(exist_ok=True)

    # ==========================================
    # PHASE 1: Train vsLSTM (Importance Only)
    # ==========================================
    print("--- Starting Phase 1: Training vsLSTM (Importance) ---")
    optimizer_phase1 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(50): # Example epochs
        model.train()
        total_loss = 0.0
        
        for seq_idx in range(len(train_set[0])):
            video = torch.tensor(train_set[0][seq_idx], dtype=torch.float32).to(device)
            gt_score = torch.tensor(train_set[1][seq_idx], dtype=torch.float32).to(device)
            
            optimizer_phase1.zero_grad()
            
            q_score, _ = model(video)
            
            # Phase 1 only cares about matching the ground truth importance scores
            loss = criterion_score(q_score.squeeze(), gt_score.squeeze())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_phase1.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Phase 1 - Epoch {epoch+1}, MSE Loss: {total_loss / len(train_set[0]):.4f}")
            
    # Save Phase 1 weights
    torch.save(model.state_dict(), save_dir / 'vsLSTM_phase1.pt')

    # ==========================================
    # PHASE 2: Train dppLSTM (Diversity)
    # ==========================================
    print("\n--- Starting Phase 2: Training dppLSTM (Diversity) ---")
    # Load the converged Phase 1 weights
    model.load_state_dict(torch.save(save_dir / 'vsLSTM_phase1.pt'))
    
    # Optional: Freeze the classification MLP so DPP only alters the kernel representations
    # for param in model.classify_mlp.parameters():
    #     param.requires_grad = False

    optimizer_phase2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
    dpp_weight = 0.1 # Tuning parameter to balance Importance vs Diversity
    
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        
        for seq_idx in range(len(train_set[0])):
            video = torch.tensor(train_set[0][seq_idx], dtype=torch.float32).to(device)
            gt_score = torch.tensor(train_set[1][seq_idx], dtype=torch.float32).to(device)
            gt_summary = torch.tensor(train_set[2][seq_idx], dtype=torch.float32).to(device)
            
            optimizer_phase2.zero_grad()
            
            q_score, pred_k = model(video)
            
            # Phase 2 combines the MSE loss with the DPP subset loss
            loss_score = criterion_score(q_score.squeeze(), gt_score.squeeze())
            loss_dpp = criterion_dpp(pred_k, gt_summary)
            
            loss = loss_score + (dpp_weight * loss_dpp)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_phase2.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Phase 2 - Epoch {epoch+1}, Total Loss: {total_loss / len(train_set[0]):.4f}")

    # Save final dppLSTM weights
    torch.save(model.state_dict(), save_dir / 'dppLSTM_final.pt')
    print("Training Complete. Final model saved.")