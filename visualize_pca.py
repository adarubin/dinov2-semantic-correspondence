import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize joint DINOv2 PCA feature maps and interactive semantic part correspondence."
    )
    parser.add_argument(
        "--img1", type=str, required=True, help="Path to the first image"
    )
    parser.add_argument(
        "--img2", type=str, required=True, help="Path to the second image"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=896,
        help="Image size (must be divisible by 14, e.g., 224, 448, 896)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vits14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        help="DINOv2 model variant",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pca_correspondence_result.png",
        help="Filename to save the comparison plot",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "bicubic", "bilinear"],
        help="Matplotlib interpolation for PCA maps",
    )
    parser.add_argument(
        "--query_x",
        type=int,
        default=None,
        help="X coordinate in patch grid space on img1 (optional)",
    )
    parser.add_argument(
        "--query_y",
        type=int,
        default=None,
        help="Y coordinate in patch grid space on img1 (optional)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive click mode to select query point on img1",
    )
    return parser.parse_args()


def load_model(model_name, device):
    print(f"Loading {model_name} from PyTorch Hub...")
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
    model.eval()
    return model


def find_correspondence(patch_tokens, query_y, query_x, grid_size):
    """Computes dense cosine similarity between a query patch in image 1

    and all patches in image 2.
    """
    # 1. Extract query patch feature from image 1: [1, 1, D]
    query_idx = query_y * grid_size + query_x
    q_vec = patch_tokens[0:1, query_idx : query_idx + 1, :]

    # 2. Extract all patch features from image 2: [1, N, D]
    target_tokens = patch_tokens[1:2, :, :]

    # 3. L2 normalize and compute cosine similarity across patches
    q_norm = F.normalize(q_vec, dim=-1)
    target_norm = F.normalize(target_tokens, dim=-1)
    similarity = torch.bmm(target_norm, q_norm.transpose(1, 2)).squeeze()  # [N]

    # 4. Find the best matching patch index in image 2
    best_idx = torch.argmax(similarity).item()
    best_y = best_idx // grid_size
    best_x = best_idx % grid_size

    sim_map = similarity.reshape(grid_size, grid_size).cpu().numpy()
    return (best_y, best_x), sim_map


def get_interactive_point(img, img_size, patch_size=14):
    """Opens an interactive window for the user to click any point on Image 1

    and maps it to patch grid coordinates.
    """
    grid_size = img_size // patch_size
    display_img = img.resize((img_size, img_size))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display_img)
    ax.set_title("Click anywhere on Image 1 to find its correspondence in Image 2")
    ax.axis("off")
    plt.tight_layout()

    print("\nPlease click on a feature in Image 1 (wheel, headlight, eye, etc.)...")
    pts = plt.ginput(n=1, timeout=0)  # timeout=0 waits indefinitely until click
    plt.close(fig)

    if not pts:
        print("No point clicked. Falling back to image center.")
        return grid_size // 2, grid_size // 2

    click_x, click_y = pts[0]
    # Map pixel coordinates to patch indices [0, grid_size - 1]
    patch_x = int(np.clip(click_x // patch_size, 0, grid_size - 1))
    patch_y = int(np.clip(click_y // patch_size, 0, grid_size - 1))

    print(f"Selected pixel: ({int(click_x)}, {int(click_y)}) -> Patch: ({patch_x}, {patch_y})")
    return patch_x, patch_y


def main():
    args = parse_args()

    # Verify that image size is divisible by the DINOv2 patch size (14)
    if args.img_size % 14 != 0:
        raise ValueError(
            f"img_size must be divisible by 14, got {args.img_size}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.model_name, device)

    # Define image transformations and normalization
    transform = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and prepare both images
    img1 = Image.open(args.img1).convert("RGB")
    img2 = Image.open(args.img2).convert("RGB")

    tensor1 = transform(img1).unsqueeze(0).to(device)
    tensor2 = transform(img2).unsqueeze(0).to(device)
    batch = torch.cat([tensor1, tensor2], dim=0)

    print("Extracting DINOv2 patch tokens...")
    with torch.no_grad():
        features = model.forward_features(batch)
        patch_tokens = features["x_norm_patchtokens"]

    B, N, D = patch_tokens.shape
    grid_size = int(N**0.5)
    patch_size = args.img_size // grid_size
    print(f"Tokens extracted: {N} patches per image ({grid_size}x{grid_size} grid)")

    # Perform joint PCA with 3 components (for RGB channels)
    print("Fitting joint PCA (3 components)...")
    tokens_flat = patch_tokens.reshape(B * N, D).cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(tokens_flat)

    # Normalize PCA values to [0, 1] range for RGB visualization
    pca_min = pca_features.min(axis=0)
    pca_max = pca_features.max(axis=0)
    pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-6)

    # Reshape features back into separate grid images for each input image
    pca_images = pca_normalized.reshape(B, grid_size, grid_size, 3)
    pca_img1 = pca_images[0]
    pca_img2 = pca_images[1]

    # Handle query point selection (Interactive Click vs CLI args)
    if args.interactive or (args.query_x is None or args.query_y is None):
        query_x, query_y = get_interactive_point(img1, args.img_size, patch_size)
    else:
        # Clamp manual coordinates to grid bounds
        query_x = max(0, min(args.query_x, grid_size - 1))
        query_y = max(0, min(args.query_y, grid_size - 1))

    print(f"Computing semantic correspondence for query patch ({query_x}, {query_y})...")
    (best_y, best_x), sim_map = find_correspondence(
        patch_tokens, query_y, query_x, grid_size
    )
    print(f"Best matching patch found in Image 2 at: ({best_x}, {best_y})")

    # Initialize subplots for visualization (3 rows x 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # Row 1: Original source images resized to match the processing resolution
    axes[0, 0].imshow(img1.resize((args.img_size, args.img_size)))
    axes[0, 0].set_title(f"Image 1: {Path(args.img1).name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2.resize((args.img_size, args.img_size)))
    axes[0, 1].set_title(f"Image 2: {Path(args.img2).name}")
    axes[0, 1].axis("off")

    # Row 2: Colored PCA feature maps
    axes[1, 0].imshow(pca_img1, interpolation=args.interpolation)
    axes[1, 0].set_title(f"PCA Map 1 ({grid_size}x{grid_size})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pca_img2, interpolation=args.interpolation)
    axes[1, 1].set_title(f"PCA Map 2 ({grid_size}x{grid_size})")
    axes[1, 1].axis("off")

    # Row 3: Part correspondence and similarity heatmap
    axes[2, 0].imshow(img1.resize((grid_size, grid_size)))
    axes[2, 0].scatter([query_x], [query_y], c="red", s=80, edgecolors="white", linewidth=1.5)
    axes[2, 0].set_title(f"Query Patch on Image 1 ({query_x}, {query_y})")
    axes[2, 0].axis("off")

    im_heat = axes[2, 1].imshow(sim_map, cmap="inferno", interpolation="bilinear")
    axes[2, 1].scatter([best_x], [best_y], c="cyan", s=80, marker="x", linewidth=2)
    axes[2, 1].set_title(f"Cosine Similarity Map on Image 2 (Best Match: {best_x}, {best_y})")
    axes[2, 1].axis("off")
    fig.colorbar(im_heat, ax=axes[2, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Visualization saved successfully to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()