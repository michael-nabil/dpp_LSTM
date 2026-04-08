import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
# Import your custom modules
from models import SummDPPLSTM
from dataset import get_dataloaders
from evaluation import generate_summary, evaluate_summary

def test_model(test_loader, model_path, nx=1024, nh=256, nout=256, device='cuda'):
    print(f"Loading trained model from {model_path}...")
    
    # Initialize model and load trained weights
    model = SummDPPLSTM(nx=nx, nh=nh, nout=nout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode (disables dropout, etc.)
    
    all_f1_scores = []
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for batch in test_loader:
            # Unpack the 7 items returned by our updated dataset.py
            video_features, _, _, change_points, n_frames, user_summary, video_id = batch
            
            # 1. Prepare data (Squeeze removes the dummy batch dimension)
            video_features = video_features.squeeze(0).to(device)
            change_points = change_points.squeeze(0).numpy()
            n_frames = n_frames.item() # Convert 1D tensor back to Python int
            user_summary = user_summary.squeeze(0).numpy()
            video_id = video_id[0]
            # video_features = F.normalize(video_features, p=2, dim=1)
            # 2. Forward Pass: Get frame-level importance scores
            q_score, _ = model(video_features)
            frame_scores = q_score.squeeze().cpu().numpy()
            
            # --- NEW FIX: Convert any negative linear outputs to 0 for Knapsack ---
            frame_scores = np.maximum(frame_scores, 0.0)

            # 3. Post-Processing: Convert frame scores to a 15% key-shot summary
            machine_summary = generate_summary(frame_scores, change_points, n_frames)
            
            # 4. Evaluation: Compare against human ground truth
            f1_score = evaluate_summary(machine_summary, user_summary)
            all_f1_scores.append(f1_score)
            
            print(f"Video: {video_id} | F1-Score: {f1_score * 100:.2f}%")
            
    # Calculate the final average F1-score across the entire test set
    mean_f1 = np.mean(all_f1_scores)
    # print(f"\n=======================================")
    # print(f"Final Average Test F1-Score: {mean_f1 * 100:.2f}%")
    # print(f"=======================================")
    return mean_f1

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path to the Kaggle file
    kaggle_h5_path = '/kaggle/input/summe-video-summarization/eccv16_dataset_summe_google_pool5.h5'
    model_weights_path = './saved_models/dppLSTM_final.pt'
    
    # We only need the test_loader for this script
    _, test_loader = get_dataloaders(kaggle_h5_path, batch_size=1)
    
    # Check if the model weights exist before testing
    if Path(model_weights_path).exists():
        test_model(test_loader, model_path=model_weights_path, device=device)
    else:
        print(f"Error: Could not find model weights at {model_weights_path}. Run train.py first!")