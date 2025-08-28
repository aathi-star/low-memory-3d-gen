Efficient 3D Point Cloud Generation: A Hybrid Approach Using LoRA, Token Attention, and Halton Sequences
This repository contains the official implementation of our research paper, currently under review at The Visual Computer journal.
Repository: https://github.com/aathi-star/low-memory-3d-gen
Clone the repository:
git clone https://github.com/aathi-star/low-memory-3d-gen.git
cd low-memory-3d-gen

Generating Point Clouds and Meshes
python generate.py \
  --checkpoint final_models/text2shape_optimized_20250601_184413_best.pt \
  --prompt "chair" \
  --output_dir final_outputs \
  --generate_mesh \
  --refinement_method poisson \
  --poisson_depth 9 \
  --guidance_scale 1.0 \
  --temperature 0.9 \
  --num_inference_steps 100 \
  --unconditional

Notes:
•	 Mesh visualization with the fallback matplotlib renderer may take ~20 seconds.
•	 Meshes are generated quickly (~2 seconds).
•	 Point clouds appear in the output folder even before mesh rendering finishes.

Training
Preprocess the dataset (ModelNet10) with Halton sequences:
python data/download_modelnet10.py --output_dir data/processed --use_halton
Train the model:
python train_optimized.py \
  --output_dir final_models \
  --batch_size 32 \
  --epochs 50 \
  --gradient_accumulation_steps 4 \
  --lora_rank 4 \
  --num_workers 16

Results
Generated Point Cloud (chair):
 
Constructed Mesh (chair):
 

Repository Status
This repository currently includes several experimental files from recent trials. The codebase will be cleaned and reorganized in the coming days for clarity and ease of use.

Citation
If you use this code, please cite our paper (citation details will be added once published).

Contact
For questions or collaboration, please open an issue in this repository.
