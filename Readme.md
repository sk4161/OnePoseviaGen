# 🎯 One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation

> **OnePoseviaGen**: A Fast and Accurate Pipeline for 6D Pose and Scale Estimation from a Single Image
<p align="center"><a href="https://arxiv.org"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://gzwsama.github.io/OnePoseviaGen.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/ZhengGeng/OnePoseviaGen'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>

---

![Teaser](assets/teaser.png)

---

Here is an improved and polished version of your TODO section in English:

---

## **TODO:**
- [ ] Train a more end-to-end model to simplify the pipeline and improve integration.
- [ ] Design a fully end-to-end network architecture to enhance the robustness and generalization capability of our model.

---

## 📄 Abstract

We introduce OnePoseviaGen, a fast and accurate pipeline for 6D pose and scale estimation from a single image. Given just one picture of an object, our system can build its 3D model from scratch and then determine its precise 3D position, orientation, and size.

---

## ⚙️ Installation

### Quick Setup Using `setup.sh`

To streamline the setup process, we provide a `setup.sh` script that automates all installation steps. Follow these instructions to get started:

### Prerequisites

Before running the `setup.sh` script, ensure you have the following prerequisites installed on your system:

- Python 3.11+
- Conda (recommended)
- Basic build tools (`git`, `make`, `cmake`, etc.)

### Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GZWSAMA/One23Pose.git
   cd One23Pose
   ```

2. **Make the `setup.sh` script executable:**

   ```bash
   chmod +x setup.sh
   ```

3. **Run the `setup.sh` script:**

   It is recommended to run this script within a fresh conda environment. Here's how you can create and activate a new environment before running the script:

   ```bash
   conda create -n one23pose python=3.11 -y
   conda activate one23pose
   ./setup.sh
   ```

   The script will handle:
   - Installing PyTorch with CUDA support.
   - Installing required dependencies.
   - Cloning and installing external extensions.
   - Building FoundationPose.
   - Installing local packages in editable mode.
   - Downloading pretrained weights.
   - Patching the transformers library.

4. **Verify the setup:**

   After running the script, verify that all dependencies are correctly installed and the necessary files are downloaded.

5. **Web demo:**

   ```bash
   python app.py
   ```
   Below is an instruction video for the usage of demo:：

   <!-- <img src="assets/instruction.gif" alt="Demo instruction" width="600"/> -->
   ![Instruction](assets/instruction.gif)

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
TODO
```

---

## 📝 Notes

- This project depends on multiple third-party libraries and modules. It is highly recommended to use a virtual environment (e.g., `conda`) for development.
- If you encounter any installation issues, refer to the official documentation of each sub-module or open an issue.
- This README will be continuously updated and improved as the project evolves.
