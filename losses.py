import torch
import torch.nn as nn

class DPPLoss(nn.Module):
    def __init__(self):
        super(DPPLoss, self).__init__()

    # FIX 1: We now pass q_score into the loss function
    def forward(self, q_score, pred_k, y_true):
        q = q_score.squeeze().double()
        pred_k = pred_k.double()

        # FIX 2: Exact Theano Math for L = K * Q
        K_mat = torch.matmul(pred_k, pred_k.T)
        Q_mat = torch.outer(q, q)
        # Build the full kernel matrix L
        L = K_mat * Q_mat # Element-wise multiplication

        I = torch.eye(L.shape[0], device=L.device).double()
        
        # Find indices where ground truth summary label is 1
        indices = torch.nonzero(y_true.squeeze() > 0.5).squeeze()
        
        # Handle cases where the ground truth subset is completely empty
        if indices.dim() == 0 or len(indices) == 0:
            return torch.tensor(0.0, device=L.device, requires_grad=True)

        # Extract sub-matrix L_Y
        L_Y = L[indices][:, indices]

        # -----------------------------------------------------
        # EXACT 2016 THEANO MATH (No slogdet, No Epsilon!)
        # -----------------------------------------------------
        det_L_Y = torch.linalg.det(L_Y)
        det_L_I = torch.linalg.det(L + I)
        
        # Original Equation: - (log(det(Ly)) - log(det(L+I)))
        # Note: In PyTorch, log(0) or log(negative) returns nan
        dpp_loss = -(torch.log(det_L_Y) - torch.log(det_L_I))
        
        # -----------------------------------------------------
        # EXACT THEANO FALLBACK (From summ_dppLSTM.py Line 114)
        # -----------------------------------------------------
        if torch.isnan(dpp_loss) or torch.isinf(dpp_loss):
            I_Y = torch.eye(L_Y.shape[0], device=L_Y.device).double()
            # Theano fallback: T.nlinalg.Det()(Ly + T.identity_like(Ly))
            fallback = torch.linalg.det(L_Y + I_Y)
            
            # 2. Final Safeguard: Protect PyTorch's backward pass from Inf gradients
            if torch.isnan(fallback) or torch.isinf(fallback):
                return torch.tensor(0.0, device=L.device, requires_grad=True)

            # This evaluates to ~1.0. When train.py calls .backward(), 
            # the gradient of this constant will be 0, perfectly 
            # matching the original 2016 codebase!
            fallback = torch.clamp(fallback, min=-3.4e38, max=3.4e38)
            return fallback.float()
        
        dpp_loss = torch.clamp(dpp_loss, min=-3.4e38, max=3.4e38)
        return dpp_loss.float()