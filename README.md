## Coding Agent Prompt — ML-Based Preprocessing for Industrial DataMatrix

### **Goal**

Implement a **production-grade ML-based preprocessing pipeline** to maximize read-rate of **industrial DataMatrix (DPM)** codes (engraved, dot-peened, wax, low-contrast) using **open-source decoders** (libdmtx / ZXing).

ML is used **only** for localization and image enhancement. Final decoding remains classical.

---

## **System Architecture**

```
Input image
  → ML segmentation (DataMatrix vs background)
  → ROI extraction + perspective normalization
  → ML image enhancement (DPM normalization)
  → Classical DataMatrix decoder
```

---

## **1. DataMatrix Localization**

### Task

* Implement **binary segmentation** model to localize DataMatrix regions.

### Requirements

* Model: lightweight **U-Net** or **SegFormer-B0**
* Input: grayscale image
* Output: binary mask (`datamatrix / background`)
* Post-processing:

  * Largest connected component
  * Contour extraction
  * Oriented bounding box
  * Perspective transform → square ROI

### Constraints

* Must handle partial, rotated, low-contrast symbols
* Single-class segmentation
* Inference < 10 ms on edge GPU

---

## **2. Geometric Normalization**

### Task

* From segmentation mask:

  * Extract contour
  * Estimate quadrilateral
  * Apply homography
  * Resize ROI to fixed resolution (e.g. 256×256)

### Constraints

* Preserve module geometry
* No cropping of finder pattern

---

## **3. ML-Based Image Enhancement**

### Task

* Implement **image-to-image CNN** that enhances DPM appearance.

### Requirements

* Model: shallow **U-Net** or **DnCNN-style residual CNN**
* Input: normalized grayscale ROI
* Output: enhanced grayscale (not RGB)
* Avoid GANs

### Training

* Synthetic + real data
* Loss:

  * L1 + SSIM
  * Edge-aware loss (Sobel)
* Optional: decoder-guided loss (binary decode success)

---

## **4. Decoder Integration**

### Task

* Integrate **libdmtx** (preferred) or **ZXing**
* Feed enhanced ROI to decoder
* Implement fallback strategy:

  * Inverted image
  * Slight contrast variants
  * Multiple thresholds

### Constraints

* Decoder remains unchanged
* No ML in decoding step

---

## **5. Synthetic Data Generation**

### Task

* Generate synthetic DPM training data:

  * Clean DataMatrix → render on metal/wax textures
  * Apply embossing, erosion, blur, lighting gradients
* Output paired samples:

  * Input: degraded
  * Target: clean

### Minimum Dataset

* 500–1k real images
* ~10k synthetic samples

---

## **6. Deployment & Performance**

### Requirements

* Training: PyTorch
* Export: ONNX
* Inference: ONNX Runtime / TensorRT
* Target runtime:

  * Segmentation: 3–6 ms
  * Enhancement: 2–5 ms
  * Decode: 1–3 ms

---

## **7. Deliverables**

1. Segmentation model + inference code
2. Enhancement model + inference code
3. ROI extraction + homography module
4. Decoder wrapper with fallback logic
5. Synthetic data generator
6. End-to-end pipeline script

---

## **Non-Goals**

* No end-to-end neural decoding
* No GAN-based hallucination
* No closed-source dependencies

---

## **Success Criteria**

* ≥2× decode-rate improvement on industrial DPM images
* Stable inference on edge devices
* Decoder accuracy improves without modification
