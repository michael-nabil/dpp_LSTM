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
        
        # Optimized PyTorch Bi-LSTM replacing the slow manual Theano loop
        self.bilstm = nn.LSTM(input_size=nx, hidden_size=nh, bidirectional=True, batch_first=False)
        
        # --- EXACT 2016 THEANO LSTM INITIALIZATION ---
        for name, param in self.bilstm.named_parameters():
            # Both weights AND biases were initialized uniformly between -0.02 and 0.02
            nn.init.uniform_(param.data, -0.02, 0.02)

        # Phase 1: Importance Scoring (vsLSTM)
        # Input size is nx (original video) + 2*nh (forward and backward hidden states)
        self.classify_mlp = MLP([nx + 2*nh, nh, 1], net_type='sigmoid')
        
        # Phase 2: Kernel Feature Extraction (dppLSTM)
        self.kernel_mlp = MLP([nx + 2*nh, nh, nout], net_type='linear')

    def forward(self, video):
        # video shape expected: (seq_len, nx)
        seq_len = video.shape[0]
        
        # 1. Initialize states from the mean visual feature
        video_mean = torch.mean(video, dim=0, keepdim=True)
        c0_base = self.c_init_mlp(video_mean) # (1, nh)
        h0_base = self.h_init_mlp(video_mean) # (1, nh)
        
        # PyTorch Bi-LSTM expects shape: (num_layers * num_directions, batch, hidden_size)
        c0 = c0_base.repeat(2, 1, 1) 
        h0 = h0_base.repeat(2, 1, 1)
        
        # 2. Run the Bi-LSTM
        video_unsq = video.unsqueeze(1) # shape: (seq_len, 1, nx)
        lstm_out, _ = self.bilstm(video_unsq, (h0, c0))
        lstm_out = lstm_out.squeeze(1)  # shape: (seq_len, 2*nh)
        
        # 3. Concatenate original features with temporal hidden states
        h_combined = torch.cat([video, lstm_out], dim=1) # (seq_len, nx + 2*nh)
        
        # 4. Phase 1 Output: Frame Importance Score (q)
        # BUG FIXED: Removing the sigmoid wrapper! The MSE loss needs the raw linear output.
        q_score = self.classify_mlp(h_combined)
        
        # 5. Phase 2 Output: Kernel Features (\phi) scaled by importance
        # BUG FIXED: Outputting raw phi, NOT pre-multiplying by q_score.
        pred_k = self.kernel_mlp(h_combined)
        
        return q_score, pred_k