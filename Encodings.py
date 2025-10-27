import torch
from torch import nn
import pandas as pd

class PatientDataEncoding(nn.Module):
    def __init__(self, nHeads, patientDataPath):
        super().__init__()
        self.ageMin = 21
        self.ageMax = 77
        self.patientDataDF = pd.read_excel(patientDataPath, sheet_name="dataset_info")

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        self.nPatientDataOutFeatures = nHeads         # needs to be divisible by nHeads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, self.nPatientDataOutFeatures)

        self.ageEncode  = lambda x: torch.tensor([(x - self.ageMin) / (self.ageMax - self.ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([0.5], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.menoEncode = lambda x: torch.tensor([1, 0]) if x == "pre" \
            else torch.tensor([0, 1]) if x == "post" \
            else torch.tensor([0, 0])
        self.densityEncode = lambda x: torch.tensor([1, 0, 0, 0]) if x == "a" \
            else torch.tensor([0, 1, 0, 0]) if x == "b" \
            else torch.tensor([0, 0, 1, 0]) if x == "c" \
            else torch.tensor([0, 0, 0, 1]) if x == "d" \
            else torch.tensor([0, 0, 0, 0])

    def forward(self, shape: tuple[int], patientIDs: list[str], device: torch.device):
        B, T, N, E, X, Y, Z = shape

        patientDataEmb = torch.empty(B, N*X*Y*Z, T, self.nPatientDataOutFeatures, device=device)      # [B*N*X*Y*Z, T, npatientDataOutFeatures]
        acqTimesTensor = torch.zeros(B, N*X*Y*Z, T, device=device)

        patientData = CleanPatientData(self.patientDataDF, patientIDs)

        for idx, md in enumerate(patientData):
            age = md['age']
            menopausal_status = md['menopause']
            breast_density = md['breast_density']
            ageEmb = self.ageEncode(age).to(device)
            menoEmb = self.menoEncode(menopausal_status).to(device)
            densityEmb = self.densityEncode(breast_density).to(device)
            concat = torch.cat((ageEmb, menoEmb, densityEmb)).unsqueeze(0)      # [1, npatientDataInFeatures]
            concat: torch.Tensor = self.patientDataEmbed(concat)                # [1, npatientDataOutFeatures]
            patientDataEmb[idx] = concat.expand(N*X*Y*Z, T, -1)                 # [B, N*X*Y*Z, T, npatientDataOutFeatures]

            acqTimes = md["acquisition_times"]
            if acqTimes is not None:
                acqTimes = acqTimes[:T]
                assert len(acqTimes) == T, f"Expected acquisition times to match num phases: {T}, got {len(acqTimes)}"
                acqTimesTensor[idx] = torch.tensor(acqTimes).expand(N*X*Y*Z, -1)

        return patientDataEmb, acqTimesTensor

# adapted from PE formula in Viswani et. al. (2023)
def PositionEncoding(seq: torch.Tensor, dim: int, div=10_000, scale=None):
    if scale is not None:
        maxVals = seq.max(dim=-1, keepdim=True).values
        seq = seq*scale / maxVals.clamp(min=1e-8)
    position = seq.unsqueeze(-1)                                # [..., N, 1]

    i = torch.arange(0, dim, 2, device=position.device)                  # [E/2]
    # divTerm = 1.0 / (div ** (i / dim))
    divTerm = torch.exp(i / dim * -torch.log(torch.tensor(div)))

    pos = position * divTerm                                   # [..., N, E/2]
    posEnc = torch.empty(*seq.shape, dim, device=position.device)       # [..., N, E]
    posEnc[..., 0::2] = torch.sin(pos)                         # [..., N, E/2] for even emb dims
    posEnc[..., 1::2] = torch.cos(pos)                         # [..., N, E/2] for odd emb dims
    
    return posEnc                                              # [..., N, E]

def PositionEncoding3D(seq: torch.Tensor, dim: int):
    # since we want the embedding dim to not be super constrained, we split the dims as evenly as we can
    # give the z-axis the short end of the stick if need be, then y-axis

    x, y, z = seq[..., 0], seq[..., 1], seq[..., 2]
    d = (dim // 2) // 3
    dX, dY, dZ = 2*d, 2*d, 2*d
    if dim % (d * 6) == 4:
        dX += 2
        dY += 2
    elif dim % (d * 6) == 2:
        dX += 2

    assert dX + dY + dZ == dim, f"You did the math wrong dummy! {dX} + {dY} + {dZ} != {dim}"
    assert not any([dX % 2, dY % 2, dZ % 2]), f"Need these dims to be even!, Got ({dX}, {dY}, {dZ})"
    
    xEnc = PositionEncoding(x, dim=dX)
    yEnc = PositionEncoding(y, dim=dY)
    zEnc = PositionEncoding(z, dim=dZ)

    posEnc = torch.cat([xEnc, yEnc, zEnc], dim=-1)
    return posEnc

# seq is tensor of indices: (T, X, Y, Z)
def PositionEncoding4D(seq: torch.Tensor, dim: int):
    assert dim % 8 == 0, f"Cannot encode 4D position unless d is divisble by 8! d: {dim}"
    t, x, y, z = seq[..., 0], seq[..., 1], seq[..., 2], seq[..., 3]
    d = dim // 4
    tEnc = PositionEncoding(t, dim=d, div=100, scale=torch.pi*2)
    xEnc = PositionEncoding(x, dim=d, div=1000)
    yEnc = PositionEncoding(y, dim=d, div=1000)
    zEnc = PositionEncoding(z, dim=d, div=1000)

    posEnc = torch.cat([tEnc, xEnc, yEnc, zEnc], dim=-1)
    return posEnc

def CleanPatientData(df: pd.DataFrame, 
                     patient_ids: list[str], 
                     columns: list[str] = ["age", 
                                           "menopause",
                                           "breast_density",
                                           "acquisition_times"]):
    def cleanMenopause(df: pd.DataFrame):
        df['menopause'] = df['menopause'].fillna('unknown')
        df['menopause'] = df['menopause'].apply(lambda x: 'pre' if 'peri' in x else x)
        df['menopause'] = df['menopause'].apply(lambda x: 'post' if 'post' in x else x)
        df['menopause'] = df['menopause'].apply(lambda x: 'pre' if 'pre' in x else x)

        return df

    df = cleanMenopause(df)

    data = []
    for pid in patient_ids:
        md = {}
        for m in columns:
            d = df.loc[df["patient_id"] == pid.upper(), m].item()
            if pd.isna(d):
                d = None
            elif m == "acquisition_times":
                d = eval(d)     # convert from string '[x, y, z]' to list [x, y, z]
            md[m] = d
        data.append(md)

    return data