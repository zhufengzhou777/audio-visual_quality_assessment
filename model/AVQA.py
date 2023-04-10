import torch
import torch.nn as nn

from model.FusedSTAttention import FusedSTBlock
from model.vit_feature_extractor import MLP


class AVQA(nn.Module):
    def __init__(self, dim=768, fusion_dim=128, length=24):
        super().__init__()
        self.attn_layer1 = FusedSTBlock(dim=dim, num_heads=8)
        self.attn_layer2 = FusedSTBlock(dim=dim, num_heads=8)
        self.regressor1 = MLP(in_features=196, hidden_features=196 // 2, out_features=1)
        self.regressor2 = MLP(in_features=196, hidden_features=196 // 2, out_features=1)
        self.anormal = nn.LayerNorm(dim)
        self.vnormal = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.aencoder = nn.Linear(dim, fusion_dim)
        self.vencoder = nn.Linear(dim, fusion_dim)
        self.adrop = nn.Dropout()
        self.vdrop = nn.Dropout()
        self.Wj_a = nn.Linear(length, length, bias=False)
        self.Wj_v = nn.Linear(length, length, bias=False)
        self.W_a = nn.Linear(length, fusion_dim, bias=False)
        self.W_v = nn.Linear(length, fusion_dim, bias=False)
        self.W_ca = nn.Linear(fusion_dim * 2, fusion_dim, bias=False)
        self.W_cv = nn.Linear(fusion_dim * 2, fusion_dim, bias=False)
        self.W_ha = nn.Linear(fusion_dim, length, bias=False)
        self.W_hv = nn.Linear(fusion_dim, length, bias=False)
        self.regressor3 = nn.Sequential(nn.LayerNorm(2*fusion_dim),
                                        nn.Linear(2*fusion_dim, fusion_dim//2),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.LayerNorm(fusion_dim//2),
                                        nn.Linear(fusion_dim//2, 1),
                                        nn.Dropout())

        self.score = nn.Sequential(
            nn.Linear(length,32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16,1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, video, audio):
        # visual-fused-st
        vst_token = video[:, :, 1:, :]  # [B,F,196,168]
        vst_feat = self.attn_layer1(vst_token)  # [B,F,196,768]
        visual_feat = vst_feat.permute(0,1,3,2)
        visual_feat = torch.squeeze(self.regressor1(visual_feat))  # [B,F,768]

        # audio-fused-st
        ast_token = audio[:, :, 1:, :]  # [B,F,196,168]
        ast_feat = self.attn_layer2(ast_token)  # [B,F,196,768]
        audio_feat = ast_feat.permute(0, 1, 3, 2)
        audio_feat = torch.squeeze(self.regressor2(audio_feat))  # [B,F,768]

        audio_feat = self.anormal(audio_feat)
        visual_feat = self.vnormal(visual_feat)

        X_a = self.aencoder(audio_feat)  # [B,F,D]
        X_v = self.vencoder(visual_feat) # [B,F,D]

        X_a = self.adrop(X_a)  # [B, F, D]
        X_v = self.vdrop(X_v)  # [B, F, D]

        J = torch.cat((X_a, X_v), dim=2)  # [B,F,2D]
        d = J.shape[-1]
        C_a = self.tanh(torch.matmul(self.Wj_a(X_a.transpose(1, 2)), J) * d ** -0.5)
        C_v = self.tanh(torch.matmul(self.Wj_v(X_v.transpose(1, 2)), J) * d ** -0.5)
        H_a = self.relu(self.W_a(X_a.transpose(1, 2)).transpose(1, 2) + self.W_ca(C_a).transpose(1, 2))
        H_v = self.relu(self.W_v(X_v.transpose(1, 2)).transpose(1, 2) + self.W_cv(C_v).transpose(1, 2))
        Xatt_a = self.W_ha(H_a.transpose(1, 2)).transpose(1, 2) + X_a
        Xatt_v = self.W_hv(H_v.transpose(1, 2)).transpose(1, 2) + X_v
        X_att = torch.cat((Xatt_v, Xatt_a), dim=2)
        score = torch.squeeze(self.regressor3(X_att))
        score = self.score(score)
        return score
