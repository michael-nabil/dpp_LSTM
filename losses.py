import torch
import torch.nn as nn

class DPPLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DPPLoss, self).__init__()
        self.epsilon = epsilon

    # FIX 1: We now pass q_score into the loss function
    def forward(self, q_score, pred_k, y_true):
        q = q_score.squeeze()

        # FIX 2: Exact Theano Math for L = K * Q
        K_mat = torch.matmul(pred_k, pred_k.T)
        Q_mat = torch.outer(q, q)
        # Build the full kernel matrix L
        L = K_mat * Q_mat # Element-wise multiplication

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
        
        # FIX 3: Bulletproof Theano NaN/Collapse Safeguard baked into the loss
        if det_L_Y <= 0 or det_L_I <= 0 or torch.isnan(det_L_Y) or torch.isnan(det_L_I):
            return torch.tensor(0.0, device=L.device, requires_grad=True)

        # Compute Negative Log-Likelihood of the DPP
        loss = torch.log(det_L_I + self.epsilon) - torch.log(det_L_Y + self.epsilon)
        return loss