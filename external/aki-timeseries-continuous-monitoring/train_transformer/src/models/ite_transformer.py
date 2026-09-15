import torch
import torch.nn as nn


class ITE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]

        self.type_emb = nn.Embedding(len(config["type_dict"]), d_model)
        self.value_emb = nn.Sequential(
            nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model, bias=False)
        )
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, 3] -> [type, time, value]
        type_emb = self.type_emb(x[..., 0].long())
        time_emb = self.time_emb(x[..., 1:2])
        value_emb = self.value_emb(x[..., 2:3])

        return type_emb + time_emb + value_emb


class ITETransformer(nn.Module):
    def __init__(self, config: dict, device: str = None):
        super().__init__()
        d_model = config["d_model"]

        # Validation
        required = ["d_model", "num_layers", "num_heads", "dropout", "num_classes"]
        if not all(k in config for k in required):
            raise ValueError(f"Missing keys: {set(required) - set(config.keys())}")
        if d_model % config["num_heads"] != 0:
            raise ValueError(f"d_model {d_model} not divisible by heads {config['num_heads']}")

        self.embedding = ITE(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(config["dropout"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["num_heads"],
            dim_feedforward=d_model * 4,
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config["num_layers"])

        # Classification head with optional pooler
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(config["dropout"]),
            nn.Linear(d_model, config["num_classes"])
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_features: bool = False
    ):
        batch_size = x.size(0)

        # Embed and add cls token
        x = self.dropout(self.embedding(x))
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply mask if provided
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            x = self.encoder(x, src_key_padding_mask=~mask)
        else:
            x = self.encoder(x)

        # Classification
        cls_features = x[:, 0]
        logits = self.classifier(cls_features)

        return (logits, cls_features) if return_features else logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = {
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "num_classes": 3,
        "type_dict": {
            "age": 0,
            "sex": 1,
            "sbp": 2,
            "dbp": 3,
            "creatinine": 4,
            "albumin": 5,
        },
    }

    model = ITETransformer(config).to(device)

    # Test data: [type, time, value]
    x = torch.cat(
        [
            torch.randint(0, 6, (4, 10, 1)),  # type
            torch.randn(4, 10, 1),  # time
            torch.randn(4, 10, 1),  # value
        ],
        dim=-1,
    ).to(device)

    mask = torch.zeros(4, 10, dtype=torch.bool).to(device)
    for i in range(4):
        mask[i, : torch.randint(1, 11, (1,)).item()] = True

    with torch.no_grad():
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=-1)

    print(f"Input: {x.shape}, Logits: {logits.shape}")
    print(f"Sample: logits={logits}, probs={probs}")
