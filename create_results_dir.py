import os

checkpoint_ids = ['03_17_18_46']

for checkpoint_id in checkpoint_ids:
    cmlf_base_results_path = os.path.join('results', 'multimodallf', checkpoint_id)

    os.makedirs(os.path.join(cmlf_base_results_path, 'output'), exist_ok=True)

    os.makedirs(os.path.join(cmlf_base_results_path, 'distance', 'vision'), exist_ok=True)
    os.makedirs(os.path.join(cmlf_base_results_path, 'distance', 'tactile'), exist_ok=True)

    os.makedirs(os.path.join(cmlf_base_results_path, 'umap', 'vision'), exist_ok=True)
    os.makedirs(os.path.join(cmlf_base_results_path, 'umap', 'tactile'), exist_ok=True)

    os.makedirs(os.path.join(cmlf_base_results_path, 'evolution_latent', 'vision'), exist_ok=True)
    os.makedirs(os.path.join(cmlf_base_results_path, 'evolution_latent', 'tactile'), exist_ok=True)

    os.makedirs(os.path.join(cmlf_base_results_path, 'evolution_aligned', 'vision'), exist_ok=True)
    os.makedirs(os.path.join(cmlf_base_results_path, 'evolution_aligned', 'tactile'), exist_ok=True)

    os.makedirs(os.path.join(cmlf_base_results_path, 'reconstruction', 'vision'), exist_ok=True)
    os.makedirs(os.path.join(cmlf_base_results_path, 'reconstruction', 'tactile'), exist_ok=True)

