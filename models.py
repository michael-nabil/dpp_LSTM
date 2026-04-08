import torch
import torch.nn as nn

class MLP(nn.Module):
    """Standard Multi-Layer Perceptron used for projections and classification."""
    def __init__(self, layers, net_type='tanh'):
        super(MLP, self).__init__()
        self.net_type = net_type
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)
        ])
        self._init_weights()
    
    def _init_weights(self):
        for linear in self.linears:
            nn.init.uniform_(linear.weight, -0.02, 0.02) 
            nn.init.zeros_(linear.bias)
            
    def forward(self, x):
        h = x
        # Hidden layers use sigmoid (as per original code)
        for linear in self.linears[:-1]:
            h = torch.sigmoid(linear(h))
        
        # Output layer activation
        out = self.linears[-1](h)
        if self.net_type == 'tanh': return torch.tanh(out)
        elif self.net_type == 'sigmoid': return torch.sigmoid(out)
        elif self.net_type == 'linear': return out
        return out

class SummDPPLSTM(nn.Module):
    """Bidirectional LSTM matching the original paper's exact concatenations."""
    def __init__(self, nx, nh, nout=256):
        super(SummDPPLSTM, self).__init__()
        self.nx = nx
        self.nh = nh
        
        # Initial state projectors (mean of video -> h0, c0)
        self.c_init_mlp = MLP([nx, nh], net_type='tanh')
        self.h_init_mlp = MLP([nx, nh], net_type='tanh')
        
        # New Change: Unidirectional LSTM to force shared weights, as implemented in the paper's source code
        # Set bidirectinoal to False
        self.lstm = nn.LSTM(input_size=nx, hidden_size=nh, bidirectional=False, batch_first=False)
        
        # --- EXACT 2016 THEANO LSTM INITIALIZATION ---
        for name, param in self.lstm.named_parameters():
            # Both weights AND biases were initialized uniformly between -0.02 and 0.02
            nn.init.uniform_(param.data, -0.02, 0.02)

        # Phase 1: Importance Scoring (vsLSTM)
        # Input size is nx (original video) + 2*nh (forward and backward hidden states)
        self.classify_mlp = MLP([nx + 2*nh, nh, 1], net_type='linear')
        
        # Phase 2: Kernel Feature Extraction (dppLSTM)
        self.kernel_mlp = MLP([nx + 2*nh, nh, nout], net_type='linear')

    def forward(self, video):
        # video shape expected: (seq_len, nx)
        seq_len = video.shape[0]
        
        # New change: Using same weights for Both LSTM directions, instead of different set of weights for each direction

        # 1. Initialize states from the mean visual feature
        video_mean = torch.mean(video, dim=0, keepdim=True)
        c0_base = self.c_init_mlp(video_mean) # (1, nh)
        h0_base = self.h_init_mlp(video_mean) # (1, nh)
        
        # Shape for unidirectional: (num_layers=1, batch=1, hidden_size)
        c0 = c0_base.unsqueeze(0) 
        h0 = h0_base.unsqueeze(0)
        
        video_unsq = video.unsqueeze(1) # shape: (seq_len, 1, nx)
        # 2. Forward Pass
        lstm_fwd, _ = self.lstm(video_unsq, (h0, c0))
        
        # 3. Backward Pass (Same LSTM, Same Weights, Reversed Input)
        video_rev = video_unsq.flip(0)
        lstm_bwd, _ = self.lstm(video_rev, (h0, c0))
        lstm_bwd = lstm_bwd.flip(0) # Un-reverse the output
        
        # 4. Concatenate
        h_combined = torch.cat([video, lstm_fwd.squeeze(1), lstm_bwd.squeeze(1)], dim=1)

        # 4. Phase 1 Output: Frame Importance Score (q)
        # BUG FIXED: Removing the sigmoid wrapper! The MSE loss needs the raw linear output.
        q_score = self.classify_mlp(h_combined)
        
        # 5. Phase 2 Output: Kernel Features (\phi) scaled by importance
        # BUG FIXED: Outputting raw phi, NOT pre-multiplying by q_score.
        pred_k = self.kernel_mlp(h_combined)
        
        return q_score, pred_k