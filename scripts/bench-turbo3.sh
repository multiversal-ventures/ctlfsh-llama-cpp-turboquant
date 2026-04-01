#!/bin/bash
# Benchmark turbo3 vs q8_0 at multiple context lengths
# Run on GCE T4: bash scripts/bench-turbo3.sh

set -e
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

MODEL="${MODEL:-/home/kartikthakore/.cuttlefish/models/gemma3-4b-it-q4_k_m.gguf}"
WIKI="${WIKI:-/tmp/wikitext-2-raw/wiki.test.raw}"
BIN="./build/bin"
RESULTS="/tmp/bench-turbo3-results.txt"

echo "=== TurboQuant CUDA Benchmark ===" | tee $RESULTS
echo "Model: $(basename $MODEL)" | tee -a $RESULTS
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a $RESULTS
echo "Date: $(date -Iseconds)" | tee -a $RESULTS
echo "" | tee -a $RESULTS

# Generation speed benchmark at different prompt lengths
echo "--- Generation Speed (tok/s) ---" | tee -a $RESULTS
printf "%-10s %-8s %-10s %-10s %-12s\n" "cache" "ctx" "prompt" "gen" "vram_MB" | tee -a $RESULTS

for CACHE in q8_0 turbo3; do
  for PROMPT_LEN in 128 512 2048; do
    # Generate a prompt of the right length by repeating text
    PROMPT=$(head -c $((PROMPT_LEN * 4)) $WIKI | tr '\n' ' ')

    # Get VRAM before
    VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    OUTPUT=$(timeout 60 $BIN/llama-cli \
      -m "$MODEL" \
      --cache-type-k $CACHE --cache-type-v $CACHE \
      -ngl 99 -n 20 \
      -p "$PROMPT" \
      --no-display-prompt -e \
      -c $((PROMPT_LEN + 128)) 2>&1 || true)

    # Extract tok/s
    PROMPT_TS=$(echo "$OUTPUT" | grep -oP 'Prompt: \K[0-9.]+' | tail -1)
    GEN_TS=$(echo "$OUTPUT" | grep -oP 'Generation: \K[0-9.]+' | tail -1)

    # Get VRAM after
    VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    printf "%-10s %-8s %-10s %-10s %-12s\n" \
      "$CACHE" "$PROMPT_LEN" "${PROMPT_TS:-err}" "${GEN_TS:-err}" "$VRAM_AFTER" | tee -a $RESULTS
  done
done

# Perplexity comparison
echo "" | tee -a $RESULTS
echo "--- Perplexity (wikitext-2, 5 chunks) ---" | tee -a $RESULTS

for CACHE in q8_0 turbo3; do
  PPL_OUT=$(timeout 180 $BIN/llama-perplexity \
    -m "$MODEL" \
    --cache-type-k $CACHE --cache-type-v $CACHE \
    -ngl 99 -f "$WIKI" --chunks 5 --ppl-stride 0 -b 512 2>&1 || true)

  PPL=$(echo "$PPL_OUT" | grep -oP 'PPL = \K[0-9.]+' || echo "err")
  PPL_ERR=$(echo "$PPL_OUT" | grep -oP '\+/- \K[0-9.]+' || echo "")

  echo "$CACHE: PPL = $PPL +/- $PPL_ERR" | tee -a $RESULTS
done

echo "" | tee -a $RESULTS
echo "--- KV Cache Size Comparison ---" | tee -a $RESULTS
echo "q8_0:   8 bits/value (1x)" | tee -a $RESULTS
echo "turbo3: 3.5 bits/value (0.44x = 4.6x compression)" | tee -a $RESULTS
echo "" | tee -a $RESULTS
echo "Results saved to $RESULTS"
