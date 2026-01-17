import torch
from torch import nn
import pandas as pd

class RiskFactorPrediction(nn.Module):
    def __init__(self, patientDataDF: pd.DataFrame, embDim: int, featureDim: int):
        super().__init__()
        self.patientDataDF = patientDataDF
        self.hiddenDim = 256
        self.outDim = featureDim

        self.pred = nn.Sequential(
            nn.LayerNorm(embDim),
            nn.Linear(embDim, self.hiddenDim),
            nn.LayerNorm(self.hiddenDim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(self.hiddenDim, self.outDim)
        )

        self.lossCat = nn.BCEWithLogitsLoss()
        self.lossCont = nn.MSELoss()

    def forward(self, latent: torch.Tensor):
        return self.pred(latent.flatten(1))      # [B, E] -> [B, 45]
    
    def maskRiskFactors(self, riskFactors: list[torch.Tensor], pred: torch.Tensor, continuousIndices: list[int] = [0]):
        mask = [(x != 0).any().item() for x in riskFactors]
        mask[0] = (riskFactors[0] != -1).any().item()       # only Age is not one-hot encoded, so look for missing value of -1
        loss = []
        out = []
        idx = 0
        for i, factor in enumerate(riskFactors):
            l = len(factor)
            x = pred[idx:idx+l]

            if mask[i] == 0:
                out.append(torch.sigmoid(x))                # use sigmoid of prediction to act as "one-hot" encoding
            else:
                out.append(factor + (x - x.detach()))       # reparameterize w.r.t. x to not break gradient flow
                if i in continuousIndices:
                    loss.append(self.lossCont(torch.sigmoid(x), factor))
                else:
                    loss.append(self.lossCat(x, factor))
            idx += l

        return torch.cat(out, dim=-1), torch.stack(loss).mean() if len(loss) else None

class PatientDataEncoding(nn.Module):
    def __init__(self, patientDataPath, embDim):
        super().__init__()
        self.ageMin = 21
        self.ageMax = 77
        self.patientDataDF = pd.read_excel(patientDataPath, sheet_name="dataset_info")

        self.inFeatures = 45             # one-hot encoding for most variables, continuous for age
        self.outFeatures = 64
        self.patientDataEmbed = nn.Linear(self.inFeatures, self.outFeatures)

        self.ageEncode  = lambda x: torch.tensor([(x - self.ageMin) / (self.ageMax - self.ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([-1], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.binaryEncode = lambda x: torch.tensor([x==0, x==1], dtype=torch.float)
        self.menoEncode = lambda x: torch.tensor([x=="pre", x=="post"], dtype=torch.float)

        nottinghamGrades = ["low", "intermediate", "high"]
        self.nottEncode = lambda x: torch.tensor([n in str(x) for n in nottinghamGrades], dtype=torch.float)
        
        densities = ["a", "b", "c", "d"]
        self.densityEncode = lambda x: torch.tensor([d in str(x) for d in densities], dtype=torch.float)
        
        tumorSubtypes = ["her2_pure", "her2_enriched", "triple negative", "luminal"]
        self.tumorEncode = lambda x: torch.tensor([t in str(x) for t in tumorSubtypes], dtype=torch.float)
        
        bmiClasses = ["underweight", "normal", "overweight", "obesity_class_1", "obesity_class_2", "obesity_class_3"]
        self.bmiEncode = lambda x: torch.tensor([b in str(x) for b in bmiClasses], dtype=torch.float)
        
        nacAgents = ["ABT 888", "AMG 386", "Anthracycline", "Carboplatin", "FEC100", "Ganetespib", "Ganitumab", "MK-2206", 
                     "Neratinib", "Paclitaxel", "Pembrolizumab", "Pertuzumab", "T-DM1", "Taxane", "Trastuzumab"]
        self.nacEncode = lambda x: torch.tensor([a in str(x) for a in nacAgents], dtype=torch.float)

        self.riskFactorPred = RiskFactorPrediction(self.patientDataDF, embDim, self.inFeatures)

    def forward(self, x: torch.Tensor, patientIDs: list[str]):
        patientData = CleanPatientData(self.patientDataDF, patientIDs, columns=[
            'age', 'anti_her2_neu_therapy', 'er', 'hr', 'pr', 'menopause', 'multifocal_cancer',
            'nottingham_grade', 'breast_density', 'tumor_subtype', 'bmi_group', 'nac_agent'
        ])

        preds = self.riskFactorPred(x)
        loss = []
        encodingTensors = []
        for idx, md in enumerate(patientData):
            age = self.ageEncode(md['age'])
            antiHER2 = self.binaryEncode(md['anti_her2_neu_therapy'])
            er = self.binaryEncode(md['er'])
            hr = self.binaryEncode(md['hr'])
            pr = self.binaryEncode(md['pr'])
            meno = self.menoEncode(md['menopause'])
            multifocal = self.binaryEncode(md['multifocal_cancer'])
            nott = self.nottEncode(md['nottingham_grade'])
            density = self.densityEncode(md['breast_density'])
            subtype = self.tumorEncode(md['tumor_subtype'])
            bmi = self.bmiEncode(md['bmi_group'])
            nac = self.nacEncode(md['nac_agent'])

            encodings = [age, antiHER2, er, hr, pr, meno, multifocal, nott, density, subtype, bmi, nac]
            encodings = [t.to(x.device) for t in encodings]

            # RISK FACTOR PREDICTION
            encodings, l = self.riskFactorPred.maskRiskFactors(encodings, preds[idx])
            if l is not None: loss.append(l)         # accumulate prediction loss
            encodingTensors.append(encodings)

        embeddings: torch.Tensor = self.patientDataEmbed(torch.stack(encodingTensors))      #[B, outFeatures]

        return embeddings, torch.stack(loss).mean() if len(loss) else 0
    
    def AcquisitionTimes(self, shape: tuple[int], patientIDs: list[str], device: torch.device):
        B, T, N, E, X, Y, Z = shape
        acqTimesTensor = torch.zeros(B, N*X*Y*Z, T, device=device)
        patientData = CleanPatientData(self.patientDataDF, patientIDs, columns=["acquisition_times"])
        for idx, md in enumerate(patientData):
            acqTimes = md["acquisition_times"]
            if acqTimes is not None:
                acqTimes = acqTimes[:T]
                assert len(acqTimes) == T, f"Expected acquisition times to match num phases: {T}, got {len(acqTimes)}"
                acqTimesTensor[idx] = torch.tensor(acqTimes).expand(N*X*Y*Z, -1)

        return acqTimesTensor


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
def PositionEncoding4D(seq: torch.Tensor, dim: int, normalize01: bool = False):
    assert dim % 8 == 0, f"Cannot encode 4D position unless d is divisble by 8! d: {dim}"
    t, x, y, z = seq[..., 0], seq[..., 1], seq[..., 2], seq[..., 3]
    d = dim // 4
    tEnc = PositionEncoding(t, dim=d, div=1000)
    xEnc = PositionEncoding(x, dim=d, div=1000)
    yEnc = PositionEncoding(y, dim=d, div=1000)
    zEnc = PositionEncoding(z, dim=d, div=1000)

    posEnc = torch.cat([tEnc, xEnc, yEnc, zEnc], dim=-1)
    if normalize01:
        return (posEnc + posEnc.min()) / (posEnc.max() - posEnc.min())
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

    if "menopause" in columns:
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