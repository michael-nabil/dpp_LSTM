import torch
import torch.nn as nn

class DPPLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DPPLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_k, y_true):
        # Build the full kernel matrix L
        L = torch.matmul(pred_k, pred_k.T)
        I = torch.eye(L.shape[0], device=L.device)
        
        # Find indices where ground truth summary label is 1
        indices = torch.nonzero(y_true.squeeze() > 0.5).squeeze()
        
        # Handle cases where the ground truth subset is completely empty
        if indices.dim() == 0 or len(indices) == 0:
            return torch.tensor(0.0, device=L.device, requires_grad=True)

        # Extract sub-matrix L_Y
        L_Y = L[indices][:, indices]
        
        # Calculate determinants with epsilon for numerical stability
        I_Y = torch.eye(L_Y.shape[0], device=L_Y.device)
        det_L_Y = torch.linalg.det(L_Y + self.epsilon * I_Y)
        det_L_I = torch.linalg.det(L + I)
        
        # Compute Negative Log-Likelihood of the DPP
        loss = torch.log(det_L_I + self.epsilon) - torch.log(det_L_Y + self.epsilon)
        return loss