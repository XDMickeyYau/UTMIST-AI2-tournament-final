#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --output=all_train.%j.out
#SBATCH --error=all_train.%j.err

# ************************************* #
# Shell script to train models to make them biased
# ************************************* #

set -e

version=8
pretrained_steps=2000000
train_steps=1000000

ent_coefs=(0 0.005)
ils=("betterbased")
self_play_mode="latest"
agent_class="SB3Agent"

for ec in "${ent_coefs[@]}"; do
  for i in "${ils[@]}"; do
    echo "Running with ent_coef=$ec, il=$i"
    sbatch --job-name="ver${version}_ec${ec}_${i}" --partition=cpu <<EOF
#!/bin/bash
#SBATCH --output="logs/ver${version}_ec${ec}_${i}_$(date +%Y%m%d_%H%M%S).%j.log"
#SBATCH --error="logs/ver${version}_ec${ec}_${i}_$(date +%Y%m%d_%H%M%S).%j.err"

python3 train.py \
  --version "${version}" \
  --pretrained_steps "${pretrained_steps}" \
  --train_steps "${train_steps}" \
  --ent_coef "${ec}" \
  --il "${i}" \
  --self_play_mode "${self_play_mode}" \
  --agent_class "${agent_class}"
EOF
  done
done