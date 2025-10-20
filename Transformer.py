import copy
import pandas as pd
import torch
import torch.nn as nn
from AttentionPooling import AttentionPooling
from helpers import PositionEncoding, PositionEncoding3D, CleanPatientData

class MyTransformerTS(nn.Module):
    def __init__(self,
                 patch_size,
                 channels,
                 nHeads,
                 nLayers,
                 patient_data_path):
        super().__init__()

        self.patch_size = patch_size

        self.emb_dim = channels[-1]

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        self.nPatientDataOutFeatures = nHeads         # needs to be divisible by nHeads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, self.nPatientDataOutFeatures)

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.cls_token   = nn.Parameter(torch.randn((1, 1, 1, self.emb_dim + self.nPatientDataOutFeatures)))
        
        ageMin = 21
        ageMax = 77
        self.ageEncode  = lambda x: torch.tensor([(x - ageMin) / (ageMax - ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([0.5], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.menoEncode = lambda x: torch.tensor([1, 0]) if x == "pre" \
            else torch.tensor([0, 1]) if x == "post" \
            else torch.tensor([0, 0])
        self.densityEncode = lambda x: torch.tensor([1, 0, 0, 0]) if x == "a" \
            else torch.tensor([0, 1, 0, 0]) if x == "b" \
            else torch.tensor([0, 0, 1, 0]) if x == "c" \
            else torch.tensor([0, 0, 0, 1]) if x == "d" \
            else torch.tensor([0, 0, 0, 0])

        tLayer = TransformerLayer(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=nHeads, dropout=0.15)
        sLayer = TransformerLayer(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=nHeads, dropout=0.15)

        self.transformerT   = Transformer(tLayer, num_layers=nLayers)
        self.temporal_proj  = AttentionPooling(self.emb_dim + self.nPatientDataOutFeatures, nHeads)
        self.transformerS   = Transformer(sLayer, num_layers=nLayers)
        self.fc_to_patches  = nn.Linear(self.emb_dim + self.nPatientDataOutFeatures, self.emb_dim)

        self._initialize_weights()

        self.patient_data_df = pd.read_excel(patient_data_path, sheet_name="dataset_info")

    def _initialize_weights(self):
        # Initialize Transformer
        def init_weights_transformer(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.transformerT.apply(init_weights_transformer)
        self.transformerS.apply(init_weights_transformer)
    
    def forward(self, x: torch.Tensor, shape: tuple[int], patientIDs: list[str], patchIndices: torch.Tensor):
        B, T, N, E, X, Y, Z = shape
        patientDataEmb = torch.empty(B, N*X*Y*Z, T, self.patientDataEmbed.out_features, device=x.device)      # [B*N*X*Y*Z, T, npatientDataOutFeatures]
        acqTimes = torch.zeros(B, N*X*Y*Z, T)

        patientData = CleanPatientData(self.patient_data_df, patientIDs)

        # print(patientData)
        for idx, md in enumerate(patientData):
            age = md['age']
            menopausal_status = md['menopause']
            breast_density = md['breast_density']
            age_emb = self.ageEncode(age).to(x.device)
            meno_emb = self.menoEncode(menopausal_status).to(x.device)
            density_emb = self.densityEncode(breast_density).to(x.device)
            concat = torch.cat((age_emb, meno_emb, density_emb)).unsqueeze(0)   # [1, npatientDataInFeatures]
            concat: torch.Tensor = self.patientDataEmbed(concat)                # [1, npatientDataOutFeatures]
            patientDataEmb[idx] = concat.expand(N*X*Y*Z, T, -1)                 # [B, N*X*Y*Z, T, npatientDataOutFeatures]

            acq_times = md["acquisition_times"]
            if acq_times is not None:
                acq_times = acq_times[:T]
                assert len(acq_times) == T, f"Expected acquisition times to match num phases: {T}, got {len(acq_times)}"
                acqTimes[idx] = torch.tensor(acq_times).expand(N*X*Y*Z, -1)

        x = x.permute(0, 2, 4, 5, 6, 1, 3)      # [B, T, N, E, X, Y, Z] -> [B, N, X, Y, Z, T, E]; put N, X, Y, Z next to each other so they can be squished
        x = x.reshape(B, -1, T, E)              # [B, N*X*Y*Z, T, E]
        x = torch.cat((x, patientDataEmb), dim=-1)  # [B, N*X*Y*Z, T, E + npatientDataOutFeatures]

        # prepend CLS token for classification prediction
        tok = self.cls_token.expand(B, -1, T, -1)
        # print(f"tok shape: {tok.shape}")
        # print(f"x shape: {x.shape}")
        x = torch.cat((tok, x), dim=1)      # [B, N*X*Y*Z + 1, T, E + npatientDataOutFeatures]

        # temporal encoding of acquisition times
        # print(f"acqTimes shape: {acqTimes.shape}")
        temporalPosEnc = PositionEncoding(acqTimes, E + self.nPatientDataOutFeatures, div=100, scale=torch.pi*2).to(x.device)
        # print(f"Temporal pos encoding: {temporalPosEnc.shape}")
        x[:, 1:] = x[:, 1:] + temporalPosEnc

        # temporal transformer first
        x = x.reshape(-1, T, E + self.nPatientDataOutFeatures)
        x = self.transformerT(x)                                                # [B*(N*X*Y*Z + 1), T, E + npatientDataOutFeatures]
        x = x.reshape(B, -1, T, E + self.nPatientDataOutFeatures)               # [B, N*X*Y*Z + 1, T, E + npatientDataOutFeatures]
        x = self.temporal_proj(x)                                               # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]
        # print(f"Shape after attention pooling: {x.shape}")
        
        spatialPosEnc = PositionEncoding3D(patchIndices, E + self.nPatientDataOutFeatures)
        # print(f"Spatial pos encoding: {spatialPosEnc.shape}")
        x[:, 1:] = x[:, 1:] + spatialPosEnc

        # then spatial transformer
        x = self.transformerS(x)            # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]
       
        x = self.fc_to_patches(x)  # [B, N*X*Y*Z + 1, E]
        cls_token = x[:, 0]   # [B, 1, E].

        # reshape to match conv shape
        x = x[:, 1:].reshape(B, N, X, Y, Z, E)
        x = x.permute(0, 1, 5, 2, 3, 4)         # (B, N, E, X, Y, Z)

        return x, cls_token
    
class MyTransformerST(nn.Module):
    def __init__(self,
                 patch_size,
                 channels,
                 nHeads,
                 nLayers,
                 patient_data_path):
        super().__init__()

        self.patch_size = patch_size

        self.emb_dim = channels[-1]

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        self.nPatientDataOutFeatures = nHeads         # needs to be divisible by nHeads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, self.nPatientDataOutFeatures)

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.cls_token   = nn.Parameter(torch.randn((1, 1, 1, self.emb_dim + self.nPatientDataOutFeatures)))
        
        ageMin = 21
        ageMax = 77
        self.ageEncode  = lambda x: torch.tensor([(x - ageMin) / (ageMax - ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([0.5], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.menoEncode = lambda x: torch.tensor([1, 0]) if x == "pre" \
            else torch.tensor([0, 1]) if x == "post" \
            else torch.tensor([0, 0])
        self.densityEncode = lambda x: torch.tensor([1, 0, 0, 0]) if x == "a" \
            else torch.tensor([0, 1, 0, 0]) if x == "b" \
            else torch.tensor([0, 0, 1, 0]) if x == "c" \
            else torch.tensor([0, 0, 0, 1]) if x == "d" \
            else torch.tensor([0, 0, 0, 0])

        tLayer = TransformerLayer(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=nHeads, dropout=0.08)
        sLayer = TransformerLayer(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=nHeads, dropout=0.08)

        self.transformerT   = Transformer(tLayer, num_layers=nLayers)
        self.temporal_proj  = AttentionPooling(self.emb_dim + self.nPatientDataOutFeatures, nHeads)
        self.transformerS   = Transformer(sLayer, num_layers=nLayers)
        self.fc_to_patches  = nn.Linear(self.emb_dim + self.nPatientDataOutFeatures, self.emb_dim)

        self._initialize_weights()

        self.patient_data_df = pd.read_excel(patient_data_path, sheet_name="dataset_info")

    def _initialize_weights(self):
        # Initialize Transformer
        def init_weights_transformer(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.transformerT.apply(init_weights_transformer)
        self.transformerS.apply(init_weights_transformer)
    
    def forward(self, x: torch.Tensor, shape: tuple[int], patientIDs: list[str], patchIndices: torch.Tensor):
        B, T, N, E, X, Y, Z = shape
        patientDataEmb = torch.empty(B, T, N*X*Y*Z, self.patientDataEmbed.out_features, device=x.device)      # [B, T, N*X*Y*Z, npatientDataOutFeatures]
        acqTimes = torch.zeros(B, N*X*Y*Z, T)

        patientData = CleanPatientData(self.patient_data_df, patientIDs)

        # print(patientData)
        for idx, md in enumerate(patientData):
            age = md['age']
            menopausal_status = md['menopause']
            breast_density = md['breast_density']
            age_emb = self.ageEncode(age).to(x.device)
            meno_emb = self.menoEncode(menopausal_status).to(x.device)
            density_emb = self.densityEncode(breast_density).to(x.device)
            concat = torch.cat((age_emb, meno_emb, density_emb)).unsqueeze(0)   # [1, npatientDataInFeatures]
            concat: torch.Tensor = self.patientDataEmbed(concat)                # [1, npatientDataOutFeatures]
            patientDataEmb[idx] = concat.expand(T, N*X*Y*Z, -1)                 # [B, T, N*X*Y*Z, npatientDataOutFeatures]

            acq_times = md["acquisition_times"]
            if acq_times is not None:
                acq_times = acq_times[:T]
                assert len(acq_times) == T, f"Expected acquisition times to match num phases: {T}, got {len(acq_times)}"
                acqTimes[idx] = torch.tensor(acq_times).expand(X*Y*Z, -1)

        x = x.permute(0, 1, 2, 4, 5, 6, 3)      # [B, T, N, E, X, Y, Z] -> [B, T, N, X, Y, Z, E]; put N, X, Y, Z next to each other so they can be squished
        x = x.reshape(B, -1, N, E)              # [B, T, N*X*Y*Z, E]
        x = torch.cat((x, patientDataEmb), dim=-1)  # [B, T, N*X*Y*Z, E + npatientDataOutFeatures]

        # prepend CLS token for classification prediction
        tok = self.cls_token.expand(B, T, -1, -1)   # [B, T, 1, E + npatientDataOutFeatures]
        # print(f"tok shape: {tok.shape}")
        # print(f"x shape: {x.shape}")
        x = torch.cat((tok, x), dim=2)              # [B, T, N*X*Y*Z + 1, E + npatientDataOutFeatures].

        spatialPosEnc = PositionEncoding3D(patchIndices, E + self.nPatientDataOutFeatures)
        # print(f"Spatial pos encoding: {spatialPosEnc.shape}")
        x[:, :, 1:] = x[:, :, 1:] + spatialPosEnc.unsqueeze(1).expand(-1, T, -1, -1)

        # first spatial transformer
        x = x.reshape(-1, N + 1, E + self.nPatientDataOutFeatures)
        x = self.transformerS(x)                    # [B*T, N*X*Y*Z + 1, E + npatientDataOutFeatures]
        x = x.reshape(B, T, N*X*Y*Z + 1, -1)        # [B, T, N*X*Y*Z + 1, E + npatientDataOutFeatures]

        # temporal encoding of acquisition times
        # print(f"acqTimes shape: {acqTimes.shape}")

        x = x.transpose(1, 2)                       # [B, N*X*Y*Z + 1, T, E + npatientDataOutFeatures]
        temporalPosEnc = PositionEncoding(acqTimes, E + self.nPatientDataOutFeatures, div=100, scale=torch.pi*2).to(x.device)       # [B, N*X*Y*Z, T, E + npatientDataOutFeatures]
        # print(f"Temporal pos encoding: {temporalPosEnc.shape}")
        x[:, 1:] = x[:, 1:] + temporalPosEnc

        # then temporal transformer
        x = x.reshape(-1, T, E + self.nPatientDataOutFeatures)                  # [B*(N*X*Y*Z + 1), T, E + npatientDataOutFeatures]
        x = self.transformerT(x)                                                # [B*(N*X*Y*Z + 1), T, E + npatientDataOutFeatures]
        x = x.reshape(B, N*X*Y*Z + 1, T, -1)                                    # [B, N*X*Y*Z + 1, T, E + npatientDataOutFeatures]
        x = self.temporal_proj(x)                                               # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]
        
        # print(f"Shape after attention pooling: {x.shape}")
       
        x = self.fc_to_patches(x)  # [B, N*X*Y*Z + 1, E]
        cls_token = x[:, 0]   # [B, E].

        # reshape to match conv shape
        x = x[:, 1:].reshape(B, N, X, Y, Z, E)
        x = x.permute(0, 1, 5, 2, 3, 4)         # (B, N, E, X, Y, Z)

        return x, cls_token

class Transformer(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self,
                 emb_dim,
                 n_heads,
                 dropout = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.d = self.emb_dim // self.n_heads

        dim_feedforward = emb_dim * 4

        self.W_qkv      = nn.Linear(emb_dim, emb_dim * 3)
        self.W_o        = nn.Linear(emb_dim, emb_dim)
        self.linear1    = nn.Linear(emb_dim, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, emb_dim)
        self.activation = nn.ReLU()

        self.norm1      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

    # Generic transformer layer that does attention and FFN on sequence S
    def forward(self, x: torch.Tensor):
        B, S, _ = x.shape
        
        qkv: torch.Tensor = self.W_qkv(x)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.d)          # [B, S, 3, H, d]
        qkv = qkv.permute(2, 0, 1, 3, 4)                # [3, B, H, S, d]
        Q, K, V = qkv[0], qkv[1], qkv[2]                # Each: [B, H, S, d]

        # print(f"\tQ shape: {Q.shape}")
        # print(f"\tK shape: {K.shape}")
        # print(f"\tV shape: {V.shape}")

        # Compute attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d ** 0.5)
        # print(f"\tScores shape: {scores.shape}")        # (B, h, S, S)

        attn = torch.softmax(scores, dim=-1)
        # print(f"\tattn shape: {attn.shape}")            # (B, h, S, S)

        context = attn @ V                              # (B, h, S, d)
        # print(f"\tcontext shape: {context.shape}")

        # Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, S, -1)
        # print(f"\tcontext shape (after concat): {context.shape}")

        O: torch.Tensor = self.W_o(context)
        # print(f"\tO shape = {O.shape}")

        x = x + O
        x = self.norm1(x)
        x = self.dropout1(x)

        # FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))    # Linear -> ReLU -> Dropout -> Linear
        ffn_out = self.dropout2(ffn_out)

        x = x + ffn_out
        x = self.norm2(x)

        return x

if __name__ == "__main__":
    dev = torch.device("cuda")
    p = torch.randint(100, (6, 32, 4), device=dev)
    emb_dim = 320

    mod = MyTransformerST(32, [2, 4])