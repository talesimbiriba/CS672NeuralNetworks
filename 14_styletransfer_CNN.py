import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
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

# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------
imsize = 256  # keep small for speed

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)  # [1, 3, H, W]
    return img.to(device, torch.float)

def show_tensor(tensor, title=None):
    image = tensor.detach().cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# VGG19 + feature extractor
# ------------------------------------------------------------
# Full VGG19 model
base_vgg = vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()

# We want outputs of several feature layers inside base_vgg.features
# In torchvision's VGG19:
#   features.0  -> conv1_1
#   features.5  -> conv2_1
#   features.10 -> conv3_1
#   features.19 -> conv4_1
#   features.21 -> conv4_2 (content)
#   features.28 -> conv5_1
return_nodes = {
    'features.0':  'conv1_1',
    'features.5':  'conv2_1',
    'features.10': 'conv3_1',
    'features.19': 'conv4_1',
    'features.21': 'conv4_2',  # content
    'features.28': 'conv5_1',
}

feature_extractor = create_feature_extractor(base_vgg, return_nodes=return_nodes)

# Layer  What it captures             Good for
# conv1_1  edges, simple textures     fine style detail
# conv2_1  more complex textures      style
# conv3_1  patterns & repetition      style
# conv4_1  mid-level textures         large-scale style
# conv5_1  global textures            very coarse style
# conv4_2  shapes, layout, semantics  content

style_layer_names   = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']  # , 'conv5_1']
content_layer_names = ['conv4_2']#, 'conv5_1']

# ------------------------------------------------------------
# Gram matrix
# ------------------------------------------------------------
def gram_matrix(feat):
    """
    feat: [B, C, H, W] -> Gram matrix [C, C]
    """
    b, c, h, w = feat.size()
    feat = feat.view(c, h * w)
    G = torch.mm(feat, feat.t())
    return G / (c * h * w)

# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------
# content_img = load_image("./CS672NeuralNetworks/figs/content.jpg")
content_img = load_image("./CS672NeuralNetworks/figs/neuschwanstein-castle.jpg")
# content_img = load_image("./CS672NeuralNetworks/figs/lion.jpg")
style_img   = load_image("./CS672NeuralNetworks/figs/style.jpg")

assert content_img.shape == style_img.shape, \
    "Use content and style images with the same size for simplicity."

# ------------------------------------------------------------
# Precompute targets (no grad here is OK)
# ------------------------------------------------------------
with torch.no_grad():
    content_feats = feature_extractor(content_img)
    style_feats   = feature_extractor(style_img)

style_grams = {name: gram_matrix(style_feats[name]) for name in style_layer_names}

# ------------------------------------------------------------
# Initialize generated image (start from content)
# ------------------------------------------------------------
generated = (content_img.clone()).requires_grad_(True).to(device)

# ------------------------------------------------------------
# Optimization with LBFGS (instead of Adam)
# ------------------------------------------------------------
content_weight = 1e2
style_weight   = 1e10

optimizer = optim.LBFGS([generated])
num_steps = 300
step = [0]  # mutable counter so closure can update it

def closure():
    optimizer.zero_grad()

    feats = feature_extractor(generated)

    # Content loss
    content_loss = 0.0
    for name in content_layer_names:
        content_loss = content_loss + torch.mean(
            (feats[name] - content_feats[name]) ** 2
        )

    # Style loss
    style_loss = 0.0
    for name in style_layer_names:
        G_gen   = gram_matrix(feats[name])
        G_style = style_grams[name]
        style_loss = style_loss + torch.mean((G_gen - G_style) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()

    if step[0] % 50 == 0 or step[0] == 0:
        print(f"Step {step[0]}/{num_steps} | "
              f"Total: {total_loss.item():.6e} | "
              f"Content: {content_loss.item():.6e} | "
              f"Style: {style_loss.item():.6e}")
    step[0] += 1
    return total_loss

while step[0] <= num_steps:
    optimizer.step(closure)
    with torch.no_grad():
        generated.clamp_(0.0, 1.0)

# ------------------------------------------------------------
# Optimization with Adam (instead of LBFGS)
# ------------------------------------------------------------

# optimizer = optim.Adam([generated], lr=2e-2)#, betas=(0.9, 0.999))
# num_steps = 300

# for step in range(1, num_steps + 1):
#     optimizer.zero_grad()

#     feats = feature_extractor(generated)

#     # Content loss
#     content_loss = 0.0
#     for name in content_layer_names:
#         content_loss = content_loss + torch.mean(
#             (feats[name] - content_feats[name]) ** 2
#         )

#     # Style loss
#     style_loss = 0.0
#     for name in style_layer_names:
#         G_gen   = gram_matrix(feats[name])
#         G_style = style_grams[name]
#         style_loss = style_loss + torch.mean((G_gen - G_style) ** 2)

#     total_loss = content_weight * content_loss + style_weight * style_loss
#     total_loss.backward()
#     optimizer.step()

#     # Clamp to keep values in a reasonable range (helps stability)
#     with torch.no_grad():
#         generated.clamp_(0.0, 1.0)


# ------------------------------------------------------------
# Show results
# ------------------------------------------------------------
show_tensor(content_img, title="Content image")
show_tensor(style_img,   title="Style image")
show_tensor(generated,   title="Stylized result")

