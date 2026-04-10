import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
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
    optimizer_phase1 = optim.Adam(model.parameters(), lr=1e-3)
    
    # Early Stopping variables
    best_val_loss_p1 = float('inf')
    patience_counter_p1 = 0
    patience_limit_p1 = 150 # Stop if no improvement for 15 epochs
    max_epochs_p1 = 100

    # The new change by using the same minibatch size from dppLSTM_main.py
    accumulation_steps = 10

    for epoch in range(max_epochs_p1):
        # Training Loop
        model.train()
        train_loss = 0.0
        optimizer_phase1.zero_grad() # Zero gradients at the START

        # PyTorch Iterator loop replaces the old Theano index loop
        for i, (video_features, gt_score, _) in enumerate(train_loader):
            # Squeeze removes the dummy batch dimension created by DataLoader
            video_features = video_features.squeeze(0).to(device)
            gt_score = gt_score.squeeze(0).to(device)

            optimizer_phase1.zero_grad() # Zero gradients at the START

            q_score, _ = model(video_features)
            
            q_score_flat = q_score.contiguous().view(-1)
            gt_score_flat = gt_score.contiguous().view(-1)

            # Phase 1 only cares about matching the ground truth importance scores
            loss = criterion_score(q_score_flat, gt_score_flat)

            # a New Change: stop dividing the loss, The gradients are too small to survive PyTorch's Adam eps.
            loss.backward() # Accumulate the raw gradients directly!
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer_phase1.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)  # Calculation of avg_train_loss stays as it is, as we calculate total_train_loss without being scaled (in train_loss variable we accumulate the original loss of each train example)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for video_features, gt_score, _ in val_loader:
                video_features = video_features.squeeze(0).to(device)
                gt_score = gt_score.squeeze(0).to(device)
                
                q_score, _ = model(video_features)
                
                q_score_flat = q_score.contiguous().view(-1)
                gt_score_flat = gt_score.contiguous().view(-1)
                # Calculate validation MSE
                loss = criterion_score(q_score_flat, gt_score_flat)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch metrics, and Checking for Early Stopping
        print(f"Phase 1 - Epoch [{epoch+1}/{max_epochs_p1}] | Train MSE(loss): {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss_p1:
            print(f"   >> Phase 1 Val loss improved from {best_val_loss_p1:.4f} to {avg_val_loss:.4f}. Saving vsLSTM!")
            best_val_loss_p1 = avg_val_loss
            patience_counter_p1 = 0
            
            # Save the best Phase 1 weights securely
            # Path(phase1_save_path).parent.mkdir(parents=True, exist_ok=True)
            # torch.save(model.state_dict(), phase1_save_path)
        else:
            patience_counter_p1 += 1
            print(f"   >> No improvement. Patience: {patience_counter_p1}/{patience_limit_p1}")
            
            if patience_counter_p1 >= patience_limit_p1:
                print(f"--- Phase 1 Early stopping triggered at epoch {epoch+1}! ---")
                break # Exit the Phase 1 training loop

    print(f"Phase 1 Complete. Best vsLSTM Validation MSE: {best_val_loss_p1:.4f}")  # Model is saved in the Early Stopping Checking
    Path(phase1_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), phase1_save_path)
    # ==========================================
    # PHASE 2: Train dppLSTM (Diversity)
    # ==========================================
    # print("\n--- Starting Phase 2: Training dppLSTM (Diversity) ---")
    # # Load the converged Phase 1 weights
    # model.load_state_dict(torch.load(phase1_save_path))
    
    # optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-3)
    # dpp_weight = 1.0
    
    # # Variables for Monitoring each epoch's validation loss, for using Early Stopping
    # best_val_loss_p2 = float('inf')
    # patience_counter_p2 = 0
    # patience_limit_p2 = 150 # Stop if no improvement for 15 epochs
    # max_epochs_p2 = 100

    # for epoch in range(max_epochs_p2):
        
    #     # Training Loop
    #     model.train()
    #     train_loss = 0.0
    #     optimizer_phase2.zero_grad() # Zero gradients at the START

    #     for i, (video_features, gt_score, gt_summary) in enumerate(train_loader):
    #         video_features = video_features.squeeze(0).to(device)
    #         gt_score = gt_score.squeeze(0).to(device)
    #         gt_summary = gt_summary.squeeze(0).to(device)

            # Normalizing the input feature vectors
    #        # video_features = F.normalize(video_features, p=2, dim=1)
    
    #         q_score, pred_k = model(video_features)
            
    #         q_score_flat = q_score.contiguous().view(-1)
    #         gt_score_flat = gt_score.contiguous().view(-1)

    #         loss_score = criterion_score(q_score_flat, gt_score_flat)
    #         loss_dpp = criterion_dpp(q_score_flat, pred_k, gt_summary)
            
    #         loss = loss_score + (dpp_weight * loss_dpp)

    #         # --- Handling the edge case of reamining training examples (videos) less that the minibatch size (accumulation_steps) ---
    #         # Find out which index the current accumulation chunk started at
    #         # chunk_start_index = i - (i % accumulation_steps)
    #         # How many videos are remaining from this start index?
    #         # remaining_videos = len(train_loader) - chunk_start_index
    #         # The actual steps is either 10, or whatever is left over!
    #         # actual_steps = min(accumulation_steps, remaining_videos)

    #         # scaled_loss = loss / actual_steps
    #         # scaled_loss.backward() # Accumulate the gradients
    #         loss.backward() # Accumulate the raw gradients directly.
    #         train_loss += loss.item()

    #         # -----------------------------------------------------
    #         # UPDATE WEIGHTS EVERY 10 VIDEOS
    #         # -----------------------------------------------------
    #         if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
    #             torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)  # Clip gradients for the accumulated batch
    #             optimizer_phase2.step()  # Take a step
                
    #             # IMMEDIATELY zero the gradients for the next 10 videos
    #             optimizer_phase2.zero_grad()
    #     avg_train_loss = train_loss / len(train_loader)  # Calculation of avg_train_loss stays as it is, as we calculate total_train_loss without being scaled (in train_loss variable we accumulate the original loss of each train example)


    #     # Validation Loop
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad(): # Disable gradient calculation for speed and memory
    #         for video_features, gt_score, gt_summary in val_loader:
    #             video_features = video_features.squeeze(0).to(device)
    #             gt_score = gt_score.squeeze(0).to(device)
    #             gt_summary = gt_summary.squeeze(0).to(device)
                
    #             q_score, pred_k = model(video_features)
                
    #             q_score_flat = q_score.contiguous().view(-1)
    #             gt_score_flat = gt_score.contiguous().view(-1)

    #             loss_score = criterion_score(q_score_flat, gt_score_flat)
    #             loss_dpp = criterion_dpp(q_score_flat, pred_k, gt_summary)
                
    #             loss = loss_score + (dpp_weight * loss_dpp)
    #             val_loss += loss.item()
                
    #     avg_val_loss = val_loss / len(val_loader)
        
    #     # Print epoch metrics, and Checking for Early Stopping

    #     print(f"Epoch [{epoch+1}/{max_epochs_p2}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    #     if avg_val_loss < best_val_loss_p2:
    #         print(f"   >> Validation loss improved from {best_val_loss_p2:.4f} to {avg_val_loss:.4f}. Saving model!")
    #         best_val_loss_p2 = avg_val_loss
    #         patience_counter_p2 = 0
            
    #         # Save the model safely
    #         # Path(final_save_path).parent.mkdir(parents=True, exist_ok=True)
    #         # torch.save(model.state_dict(), final_save_path)
    #     else:
    #         patience_counter_p2 += 1
    #         print(f"   >> No improvement. Patience: {patience_counter_p2}/{patience_limit_p2}")
            
    #         if patience_counter_p2 >= patience_limit_p2:
    #             print(f"--- Early stopping triggered at epoch {epoch+1}! ---")
    #             break # Exit the training loop early

    # print(f"Phase 2 Complete. Best Validation Loss: {best_val_loss_p2:.4f}")
    # Path(final_save_path).parent.mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), final_save_path)
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path to the Kaggle file (Change this to test TVSum, OVP, or YouTube!)
    kaggle_h5_path = '/kaggle/input/summe-video-summarization/eccv16_dataset_summe_google_pool5.h5'
    
    print(f"Loading data from {kaggle_h5_path}...")
    train_loader, test_loader = get_dataloaders(kaggle_h5_path, batch_size=1)
    
    train_two_phase(train_loader, device=device)