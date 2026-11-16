#!/usr/bin/env bash
# conda 환경 'xai' 생성 + 사전학습 가중치 미리 받기.
# ⚠️ master 노드 금지. srun으로 GPU 노드를 받은 뒤, 저장소 루트에서 실행:
#   bash scripts/setup_env.sh
set -eo pipefail   # -u는 conda activate 스크립트와 충돌해서 뺌

CONDA_ROOT=/data/$USER/anaconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '^xai\s'; then
  # 기본 채널은 이용약관(ToS) 동의가 필요해서 conda-forge 사용
  conda create -y -n xai -c conda-forge --override-channels python=3.10
fi
conda activate xai

# GPU를 할당받은 상태에서 설치해야 CUDA 버전이 깔린다 (RTX 3090 -> cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 가중치 캐시를 home 대신 NAS 개인 폴더에 (home에는 설정 파일만)
export TORCH_HOME=/data/$USER/cache/torch
mkdir -p "$TORCH_HOME"
python - <<'EOF'
import torch
from torchvision import models
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("pretrained weights cached")
EOF
