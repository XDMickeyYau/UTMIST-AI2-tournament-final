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

ent_coefs=(0.005)
ils=("il")
agent_class="SB3AgentRuled"
iter=5
video=""

for ec in "${ent_coefs[@]}"; do
  for i in "${ils[@]}"; do
    echo "Running with ent_coef=$ec, il=$i"
    sbatch --job-name="ver${version}_ec${ec}_il${i}" --partition=cpu <<EOF
#!/bin/bash
#SBATCH --output="logs/ver${version}_ec${ec}_${i}_$(date +%Y%m%d_%H%M%S).%j.log"
#SBATCH --error="logs/ver${version}_ec${ec}_${i}_$(date +%Y%m%d_%H%M%S).%j.err"

python3 evaluation.py \
  --version "${version}" \
  --pretrained_steps "${pretrained_steps}" \
  --ent_coef "${ec}" \
  --il "${i}" \
  --agent_class "${agent_class}" \
  --iter ${iter} \
  ${video}
EOF
  done
done