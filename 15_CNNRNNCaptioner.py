"""
Tiny image-to-text example using an RNN decoder (CNN encoder + GRU)

Task:
    Image (single colored shape in simple position)
        -> caption like: "red circle left"

Shapes:   circle, square, triangle
Colors:   red, blue
Positions:left, center, right

Caption format:
    <bos> color shape position <eos>

Vocabulary size: small (pad, bos, eos, 2 colors, 3 shapes, 3 positions)
    pad: padding token to make all sentences with the same lenght
    bos: beginning of sentence token
    eos: end of sentence token

Dependencies:
    pip install torch torchvision pillow matplotlib
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Device selection: MPS (Mac) -> CUDA -> CPU
# ---------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU via MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


device = get_device()


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
class Vocab:
    def __init__(self):
        # special tokens
        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

        self.colors = ["red", "blue"]
        self.shapes = ["circle", "square", "triangle"]
        self.positions = ["left", "center", "right"]

        tokens = [self.PAD, self.BOS, self.EOS] + \
                 self.colors + self.shapes + self.positions

        # token to index and index to token mappings
        self.stoi = {tok: i for i, tok in enumerate(tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    def encode_caption(self, color, shape, position):
        # <bos> color shape position <eos>
        tokens = [self.BOS, color, shape, position, self.EOS]
        return torch.tensor([self.stoi[t] for t in tokens], dtype=torch.long)

    def decode_tokens(self, token_ids):
        toks = []
        for i in token_ids:
            tok = self.itos[int(i)]
            if tok == self.EOS:
                break
            if tok not in (self.BOS, self.PAD):
                toks.append(tok)
        return " ".join(toks)

    @property
    def pad_idx(self):
        return self.stoi[self.PAD]

    @property
    def bos_idx(self):
        return self.stoi[self.BOS]

    @property
    def eos_idx(self):
        return self.stoi[self.EOS]

    @property
    def size(self):
        return len(self.stoi)


vocab = Vocab()


# ---------------------------------------------------------------------
# Synthetic dataset: shapes rendered with PIL
# ---------------------------------------------------------------------
class ShapesCaptionDataset(Dataset):
    """
    Generates synthetic images + captions.

    Each sample: (image, caption_tensor)
        image: 3xH×W float tensor in [0,1]
        caption: LongTensor of shape (5,)  <bos> color shape position <eos>
    """

    def __init__(self, num_samples=5000, image_size=64, seed=0):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = random.Random(seed)

        self.colors = vocab.colors
        self.shapes = vocab.shapes
        self.positions = vocab.positions

        # Pre-generate label triplets to keep dataset deterministic
        self.triplets = []
        for _ in range(num_samples):
            c = self.rng.choice(self.colors)
            s = self.rng.choice(self.shapes)
            p = self.rng.choice(self.positions)
            self.triplets.append((c, s, p))

    def __len__(self):
        return self.num_samples

    def _rgb(self, color):
        return {"red": (220, 50, 50), "blue": (50, 80, 220)}[color]

    def _shape_bbox(self, position, size, margin=8):
        """Return bounding box (left, top, right, bottom) for the shape."""
        W = H = size
        w = h = size // 3  # size of the shape

        if position == "left":
            cx = W // 4
        elif position == "center":
            cx = W // 2
        else:  # right
            cx = 3 * W // 4

        cy = H // 2
        left = cx - w // 2
        top = cy - h // 2
        right = cx + w // 2
        bottom = cy + h // 2

        # clamp to margins
        left = max(margin, left)
        top = max(margin, top)
        right = min(W - margin, right)
        bottom = min(H - margin, bottom)
        return left, top, right, bottom

    def _draw_shape(self, draw, shape, bbox, color):
        if shape == "circle":
            draw.ellipse(bbox, fill=color)
        elif shape == "square":
            draw.rectangle(bbox, fill=color)
        elif shape == "triangle":
            left, top, right, bottom = bbox
            cx = (left + right) // 2
            points = [(cx, top), (left, bottom), (right, bottom)]
            draw.polygon(points, fill=color)

    def __getitem__(self, idx):
        color, shape, position = self.triplets[idx]

        # Create RGB image with light background
        img = Image.new("RGB", (self.image_size, self.image_size),
                        color=(245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Draw shape
        bbox = self._shape_bbox(position, self.image_size)
        self._draw_shape(draw, shape, bbox, self._rgb(color))

        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0  # HxWx3
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # 3xHxW

        # Caption
        caption = vocab.encode_caption(color, shape, position)  # (5,)

        return img_tensor, caption


# ---------------------------------------------------------------------
# Model: CNN encoder + GRU decoder
# ---------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B,3,H,W)
        h = self.conv(x)         # (B,128,1,1)
        h = h.view(h.size(0), -1)  # (B,128)
        z = self.fc(h)           # (B,out_dim)
        return z


class RNNDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size, emb_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab.pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions_in):
        """
        features: (B, feature_dim) encoder output
        captions_in: (B, T_in) input tokens (<bos> ... last-1)
        Returns:
            logits: (B, T_in, vocab_size)
        """
        embedded = self.embedding(captions_in)  # (B,T_in,emb_dim)
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1,B,H)
        out, _ = self.gru(embedded, h0)  # (B,T_in,H)
        logits = self.fc_out(out)        # (B,T_in,V)
        return logits

    def generate(self, features, max_len=6):
        """
        Greedy decoding: start with <bos>, stop at <eos> or max_len.
        features: (B, feature_dim)
        Returns list of lists of token ids (no padding).
        """
        B = features.size(0)
        h = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1,B,H)
        inputs = torch.full((B, 1), vocab.bos_idx, dtype=torch.long, device=features.device)
        generated = [[] for _ in range(B)]

        for _ in range(max_len):
            emb = self.embedding(inputs)  # (B,1,emb_dim)
            out, h = self.gru(emb, h)     # (B,1,H)
            logits = self.fc_out(out[:, -1, :])  # (B,V)
            next_tokens = torch.argmax(logits, dim=-1)  # (B,)
            inputs = next_tokens.unsqueeze(1)

            for i in range(B):
                generated[i].append(int(next_tokens[i].item()))

        return generated


class CNNRNNCaptioner(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=feature_dim)
        self.decoder = RNNDecoder(feature_dim=feature_dim,
                                  hidden_dim=hidden_dim,
                                  vocab_size=vocab.size)

    def forward(self, images, captions):
        """
        Teacher-forced training.

        captions: (B, T) with tokens [<bos>, w1, w2, w3, <eos>]
        We use captions[:, :-1] as input and captions[:, 1:] as targets.
        """
        features = self.encoder(images)
        captions_in = captions[:, :-1]    # (B,4), get all but last token
        targets = captions[:, 1:]         # (B,4), get all but first token

        logits = self.decoder(features, captions_in)  # (B,4,V)

        return logits, targets

    def generate(self, images, max_len=6):
        # Greedy decoding, it generates list of token id lists from images
        with torch.no_grad():
            features = self.encoder(images)
            seqs = self.decoder.generate(features, max_len=max_len)
        return seqs


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0

    for imgs, caps in loader:
        imgs = imgs.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()
        logits, targets = model(imgs, caps)  # logits: (B,T,V), targets: (B,T)
        B, T, V = logits.size()
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        n += B

    return total_loss / n


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for imgs, caps in loader:
            imgs = imgs.to(device)
            caps = caps.to(device)
            logits, targets = model(imgs, caps)
            B, T, V = logits.size()

            loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
            total_loss += loss.item() * B
            n += B

            # token-level accuracy (just for a rough idea)
            preds = logits.argmax(dim=-1)  # (B,T)
            correct += (preds == targets).sum().item()
            total_tokens += targets.numel()

    return total_loss / n, correct / total_tokens


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def show_examples(model, dataset, num_examples=6):
    model.eval()
    idxs = np.random.choice(len(dataset), size=num_examples, replace=False)
    imgs = []
    true_caps = []
    for i in idxs:
        img, cap = dataset[i]
        imgs.append(img)
        true_caps.append(vocab.decode_tokens(cap))

    imgs_tensor = torch.stack(imgs, dim=0).to(device)
    gen_token_seqs = model.generate(imgs_tensor, max_len=6)

    pred_caps = [vocab.decode_tokens(seq) for seq in gen_token_seqs]

    # Plot
    cols = min(3, num_examples)
    rows = int(math.ceil(num_examples / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(num_examples):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"T: {true_caps[i]}\nP: {pred_caps[i]}", fontsize=9)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def print_caption_examples(dataset, num_examples=5, show_images=True):
    """
    Print a few caption examples from the ShapesCaptionDataset.
    Shows:
        - token IDs
        - token strings (including <bos> and <eos>)
        - decoded caption
        - optionally the image
    """
    idxs = np.random.choice(len(dataset), size=num_examples, replace=False)

    for idx in idxs:
        img, cap = dataset[idx]      # img: (3,H,W), cap: (5,)
        cap_ids = cap.tolist()
        cap_tokens = [vocab.itos[i] for i in cap_ids]
        decoded = vocab.decode_tokens(cap_ids)

        print(f"\n--- Example {idx} ---")
        print(" Token IDs:     ", cap_ids)
        print(" Token strings: ", cap_tokens)
        print(" Decoded text:  ", decoded)

        if show_images:
            plt.figure()
            plt.imshow(img.permute(1,2,0).numpy())
            plt.axis("off")
            plt.title(f"{decoded}")
            plt.show()



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters
    image_size = 64
    train_samples = 64*100
    val_samples = 1000
    batch_size = 64
    epochs = 20
    lr = 1e-3

    # Datasets
    train_ds = ShapesCaptionDataset(num_samples=train_samples, image_size=image_size, seed=0)
    val_ds = ShapesCaptionDataset(num_samples=val_samples, image_size=image_size, seed=1)

    # print caption examples: 
    # print_caption_examples(train_ds, num_examples=5)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss
    model = CNNRNNCaptioner(feature_dim=128, hidden_dim=128).to(device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} "
              f"| val loss {val_loss:.4f} | val token-acc {val_acc:.3f}")

    # Show some qualitative examples
    show_examples(model, val_ds, num_examples=6)
