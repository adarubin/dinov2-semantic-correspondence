# DINOv2 Spatial Representations & Dense Semantic Correspondence

An implementation of unsupervised spatial representation learning and zero-shot part correspondence using **DINOv2** (Vision Transformer foundation model). 

Without any fine-tuning, task-specific supervision, or manual keypoint labels, frozen DINOv2 patch tokens encode rich geometric and semantic structures capable of:
1. **Unsupervised Part Segmentation:** Projecting cross-image semantics into a shared visual subspace via Joint Principal Component Analysis (PCA).
2. **Dense Semantic Correspondence:** Accurately localizing fine-grained semantic parts (wheels, windows, eyes, muzzle) across distinct domains, poses, and illumination conditions via dense Cosine Similarity in deep feature space.

---

## Qualitative Results

### 1. Mechanical Component Matching (Wheels & Chassis)
Dense feature correspondence transfers directly from stylized animation assets to realistic sports cars.

| Wheel Selection & Correspondence | Windshield / Roof Selection |
| :---: | :---: |
| ![Wheel Match](assets/wheel_match.jpg) | ![Window Match](assets/window_match.jpg) |

---

### 2. Anatomical Keypoint Transfer (Canine Features)
Semantic feature alignment across distinct dog breeds, varying scales, dynamic poses, and motion blur without task-specific training.

| Ocular Keypoint Transfer (Eyes) | Muzzle / Tongue Alignment |
| :---: | :---: |
| ![Eyes Match](assets/eyes_match.jpg) | ![Tongue Match](assets/tongue_match.jpg) |

---

## Technical Details

1. **Backbone Feature Extraction:** Images are resized to dimensions divisible by the patch size ($14 \times 14$) and passed through a frozen DINOv2 Vision Transformer (`dinov2_vits14`) to extract normalized patch tokens $X \in \mathbb{R}^{B \times N \times D}$.
2. **Joint Dimensionality Reduction:** Patch tokens from both images are flattened and jointly fitted using **3-component PCA**. Projecting the first three principal components to RGB color channels creates consistent semantic color palettes across both frames.
3. **Dense Part Retrieval:** A query patch $q \in \mathbb{R}^{1 \times D}$ from Image 1 is compared against all target image patches $K \in \mathbb{R}^{N \times D}$ using $L_2$-normalized Cosine Similarity:
   $$\text{Sim}(q, K) = \frac{q}{\|q\|_2} \cdot \left(\frac{K}{\|K\|_2}\right)^T$$
   The target patch with the maximal similarity score represents the semantic counterpart.

---

## Installation & Setup

```bash
git clone [https://github.com/adarubin/dinov2-semantic-correspondence.git](https://github.com/adarubin/dinov2-semantic-correspondence.git)
cd dinov2-semantic-correspondence
pip install -r requirements.txt

Usage
Interactive Click Mode
Run the script without coordinates to open an interactive window where you can click any point on Image 1:

Bash
python visualize_pca.py --img1 "example pics/car1.png" --img2 "example pics/car2.png"

Scripted Execution with Fixed Patch Coordinates
Bash
# Evaluate wheel correspondence on vehicle pair
python visualize_pca.py --img1 "example pics/car1.png" --img2 "example pics/car2.png" --query_x 23 --query_y 53 --output "assets/wheel_match.jpg"

# Evaluate eye correspondence on canine pair
python visualize_pca.py --img1 "example pics/dog1.png" --img2 "example pics/dog2.png" --query_x 29 --query_y 15 --output "assets/eyes_match.jpg"
