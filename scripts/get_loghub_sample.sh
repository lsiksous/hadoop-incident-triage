#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
if [ ! -d "Loghub" ]; then
  git clone --depth 1 https://github.com/logpai/loghub.git Loghub
fi

# Exemple: copie un dataset HDFS/YARN (les dossiers exacts peuvent varier selon le repo)
# On copie tout ce qui ressemble à du hadoop/hdfs/yarn dans data/
find Loghub -maxdepth 3 -type f \( -iname "*hdfs*" -o -iname "*yarn*" -o -iname "*hadoop*" -o -iname "*.log" -o -iname "*.txt" \) \
  | head -n 20 \
  | while read -r f; do
      bn="$(basename "$f")"
      cp "$f" "data/${bn}" || true
    done

echo "OK: fichiers copiés dans ./data/"
ls -lh data | head