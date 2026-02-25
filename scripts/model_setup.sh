#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Familiar Model Setup — shared by run.sh (standalone & Mother)
# Sources into the calling script. Expects these vars to be set:
#   TOTAL_RAM_MB  — system RAM in MB (0 if unknown)
#   IS_PI         — "true" if Raspberry Pi detected
#
# After sourcing, these vars are set:
#   OLLAMA_MODEL            — main conversation model
#   FAMILIAR_LIGHTWEIGHT_MODEL — background task model (may be empty)
#   MODELS_INSTALLED        — "true" if any models were set up
# ═══════════════════════════════════════════════════════════════

MODELS_INSTALLED=false

# ── Model catalog ──
# Format: name|disk_size_label|ram_at_runtime_mb|description
declare -a ALL_MODELS=(
    "smollm2:135m|80 MB|150|Tiny — extremely constrained devices"
    "smollm2:360m|200 MB|300|Light — basic tasks, very low RAM"
    "qwen2.5:0.5b|400 MB|400|Good — best quality/size ratio"
    "tinyllama|700 MB|700|Solid — capable general purpose"
    "qwen2.5:1.5b|1.0 GB|1000|Very good — strong instruction following"
    "smollm2:1.7b|1.0 GB|1100|Very good — alternative to Qwen 1.5b"
    "llama3.2|2.0 GB|2000|Great — Meta's latest compact model"
    "qwen2.5:3b|2.0 GB|2100|Great — strong reasoning"
    "mistral|4.0 GB|4200|Excellent — Mistral 7B"
    "qwen2.5:7b|5.0 GB|5000|Excellent — best mid-size quality"
    "deepseek-r1:14b|9.0 GB|9000|Outstanding — best local reasoning"
)

get_model_field() {
    local entry="$1" field="$2"
    echo "$entry" | cut -d'|' -f"$field"
}

# ── Bundle definitions per hardware tier ──
setup_bundles() {
    if $IS_PI || [[ $TOTAL_RAM_MB -gt 0 && $TOTAL_RAM_MB -lt 4000 ]]; then
        HW_TIER="low"
        BUNDLE_FULL_MAIN="qwen2.5:0.5b"
        BUNDLE_FULL_EMBED="nomic-embed-text"
        BUNDLE_FULL_LABEL="Full Setup"
        BUNDLE_FULL_DESC="qwen2.5:0.5b (400 MB) + nomic-embed-text (137 MB)"
        BUNDLE_FULL_DETAIL="Conversation + semantic memory search"
        BUNDLE_FULL_BG=""  # Main model is small enough to double as background

        BUNDLE_MIN_MAIN="qwen2.5:0.5b"
        BUNDLE_MIN_LABEL="Minimal Setup"
        BUNDLE_MIN_DESC="qwen2.5:0.5b (400 MB)"
        BUNDLE_MIN_DETAIL="One model, lowest memory footprint"

    elif [[ $TOTAL_RAM_MB -gt 0 && $TOTAL_RAM_MB -lt 8000 ]]; then
        HW_TIER="mid"
        BUNDLE_FULL_MAIN="llama3.2"
        BUNDLE_FULL_BG="qwen2.5:0.5b"
        BUNDLE_FULL_EMBED="nomic-embed-text"
        BUNDLE_FULL_LABEL="Full Setup"
        BUNDLE_FULL_DESC="llama3.2 (2 GB) + qwen2.5:0.5b (400 MB) + nomic-embed-text (137 MB)"
        BUNDLE_FULL_DETAIL="Main model + background tasks + semantic search"

        BUNDLE_MIN_MAIN="llama3.2"
        BUNDLE_MIN_LABEL="Minimal Setup"
        BUNDLE_MIN_DESC="llama3.2 (2 GB)"
        BUNDLE_MIN_DETAIL="One model handles everything"

    else
        HW_TIER="high"
        BUNDLE_FULL_MAIN="llama3.2"
        BUNDLE_FULL_BG="qwen2.5:0.5b"
        BUNDLE_FULL_EMBED="nomic-embed-text"
        BUNDLE_FULL_LABEL="Full Setup"
        BUNDLE_FULL_DESC="llama3.2 (2 GB) + qwen2.5:0.5b (400 MB) + nomic-embed-text (137 MB)"
        BUNDLE_FULL_DETAIL="Main model + background tasks + semantic search"

        BUNDLE_MIN_MAIN="llama3.2"
        BUNDLE_MIN_LABEL="Minimal Setup"
        BUNDLE_MIN_DESC="llama3.2 (2 GB)"
        BUNDLE_MIN_DETAIL="One model handles everything"
    fi
}

# ── Custom multi-select ──
run_custom_select() {
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  Custom Model Selection                             │"
    echo "  │  Enter model numbers separated by spaces.           │"
    echo "  │  Ollama loads one model at a time, so disk space    │"
    echo "  │  is the real constraint, not RAM.                   │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo ""

    # Filter models appropriate for this hardware
    local idx=1
    local -a viable_models=()
    local -a viable_names=()

    for entry in "${ALL_MODELS[@]}"; do
        local name=$(get_model_field "$entry" 1)
        local disk=$(get_model_field "$entry" 2)
        local ram=$(get_model_field "$entry" 3)
        local desc=$(get_model_field "$entry" 4)

        # Show all models but mark ones that are tight on RAM
        local marker="  "
        if [[ $TOTAL_RAM_MB -gt 0 && $ram -gt $TOTAL_RAM_MB ]]; then
            marker="⚠ "
        fi

        printf "  [%2d] ${marker}%-20s %8s   %s\n" "$idx" "$name" "$disk" "$desc"
        viable_models+=("$entry")
        viable_names+=("$name")
        idx=$((idx + 1))
    done

    echo ""
    printf "  [%2d]   %-20s %8s   %s\n" "$idx" "nomic-embed-text" "137 MB" "Embedding model for semantic search"
    local EMB_IDX=$idx

    echo ""
    if [[ $TOTAL_RAM_MB -gt 0 ]]; then
        echo "  ⚠  = may be tight on your ${TOTAL_RAM_MB} MB RAM (but still works — Ollama swaps models)"
    fi
    echo ""
    echo "  Example: \"1 3 ${EMB_IDX}\" downloads smollm2:135m, qwen2.5:0.5b, and nomic-embed-text"
    echo ""
    read -p "  Enter numbers (space-separated): " SELECTIONS

    if [[ -z "$SELECTIONS" ]]; then
        echo "  No models selected."
        return 1
    fi

    # Parse selections
    local -a selected_models=()
    local want_embed=false
    local total_disk_label=""

    for num in $SELECTIONS; do
        if [[ "$num" == "$EMB_IDX" ]]; then
            want_embed=true
        elif [[ $num -ge 1 && $num -le ${#viable_names[@]} ]]; then
            local sel_idx=$((num - 1))
            selected_models+=("${viable_names[$sel_idx]}")
        fi
    done

    if [[ ${#selected_models[@]} -eq 0 && "$want_embed" == "false" ]]; then
        echo "  No valid models selected."
        return 1
    fi

    # Download selected models
    echo ""
    for model in "${selected_models[@]}"; do
        echo "  Downloading $model..."
        ollama pull "$model"
        echo ""
    done

    if $want_embed; then
        echo "  Downloading nomic-embed-text..."
        ollama pull nomic-embed-text
        echo ""
    fi

    # ── Auto-assign roles ──
    if [[ ${#selected_models[@]} -ge 2 ]]; then
        # Multiple models: largest = main, smallest = lightweight
        # Sort by RAM (field 3 in catalog)
        local largest="" smallest=""
        local largest_ram=0 smallest_ram=999999

        for sel_name in "${selected_models[@]}"; do
            for entry in "${ALL_MODELS[@]}"; do
                local name=$(get_model_field "$entry" 1)
                local ram=$(get_model_field "$entry" 3)
                if [[ "$name" == "$sel_name" ]]; then
                    if [[ $ram -gt $largest_ram ]]; then
                        largest_ram=$ram
                        largest="$name"
                    fi
                    if [[ $ram -lt $smallest_ram ]]; then
                        smallest_ram=$ram
                        smallest="$name"
                    fi
                fi
            done
        done

        export OLLAMA_MODEL="$largest"
        if [[ "$largest" != "$smallest" ]]; then
            export FAMILIAR_LIGHTWEIGHT_MODEL="$smallest"
            echo "  ✓ Main model:       $largest"
            echo "  ✓ Background model: $smallest"
        else
            echo "  ✓ Model: $largest"
        fi
    elif [[ ${#selected_models[@]} -eq 1 ]]; then
        export OLLAMA_MODEL="${selected_models[0]}"
        echo "  ✓ Model: ${selected_models[0]}"
    fi

    if $want_embed; then
        echo "  ✓ Semantic search: nomic-embed-text"
    fi

    MODELS_INSTALLED=true
    return 0
}

# ── Write .env file ──
write_env() {
    local env_file="$1"
    local wrote=false

    # Only write if we have something to persist
    if [[ -n "$OLLAMA_MODEL" || -n "$FAMILIAR_LIGHTWEIGHT_MODEL" ]]; then
        # Append to existing or create new
        {
            echo ""
            echo "# Generated by Familiar model setup — $(date -Iseconds)"
            echo "DEFAULT_PROVIDER=ollama"
            if [[ -n "$OLLAMA_MODEL" ]]; then
                echo "OLLAMA_MODEL=$OLLAMA_MODEL"
            fi
            if [[ -n "$FAMILIAR_LIGHTWEIGHT_MODEL" ]]; then
                echo "FAMILIAR_LIGHTWEIGHT_MODEL=$FAMILIAR_LIGHTWEIGHT_MODEL"
            fi
        } >> "$env_file"
        wrote=true
    fi

    if $wrote; then
        echo ""
        echo "  ✓ Saved to $env_file"
    fi
}

# ═══════════════════════════════════════════════════════════════
# Main model setup flow
# ═══════════════════════════════════════════════════════════════
run_model_setup() {
    setup_bundles

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  No AI models installed. Let's set you up."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ $TOTAL_RAM_MB -gt 0 ]]; then
        local ram_display
        if [[ $TOTAL_RAM_MB -ge 1024 ]]; then
            ram_display="$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_MB/1024}") GB"
        else
            ram_display="${TOTAL_RAM_MB} MB"
        fi
        echo "  System: ${ram_display} RAM"
        if $IS_PI; then echo "  Device: Raspberry Pi"; fi
    fi

    echo ""
    echo "  [1] $BUNDLE_FULL_LABEL          ⭐ Recommended"
    echo "      $BUNDLE_FULL_DESC"
    echo "      $BUNDLE_FULL_DETAIL"
    echo ""
    echo "  [2] $BUNDLE_MIN_LABEL"
    echo "      $BUNDLE_MIN_DESC"
    echo "      $BUNDLE_MIN_DETAIL"
    echo ""
    echo "  [3] Custom"
    echo "      Pick individual models from the full catalog"
    echo ""
    echo "  [4] Skip — I'll use a cloud API key instead"
    echo ""
    read -p "  Choose [1-4] (default: 1): " BUNDLE_CHOICE
    BUNDLE_CHOICE=${BUNDLE_CHOICE:-1}

    case "$BUNDLE_CHOICE" in
        1)
            # Full bundle
            echo ""
            echo "  Downloading $BUNDLE_FULL_MAIN..."
            ollama pull "$BUNDLE_FULL_MAIN"

            if [[ -n "$BUNDLE_FULL_BG" ]]; then
                echo ""
                echo "  Downloading $BUNDLE_FULL_BG (background tasks)..."
                ollama pull "$BUNDLE_FULL_BG"
                export FAMILIAR_LIGHTWEIGHT_MODEL="$BUNDLE_FULL_BG"
            fi

            if [[ -n "$BUNDLE_FULL_EMBED" ]]; then
                echo ""
                echo "  Downloading $BUNDLE_FULL_EMBED (semantic search)..."
                ollama pull "$BUNDLE_FULL_EMBED"
            fi

            export OLLAMA_MODEL="$BUNDLE_FULL_MAIN"
            MODELS_INSTALLED=true

            echo ""
            echo "  ✓ Main model:       $BUNDLE_FULL_MAIN"
            if [[ -n "$BUNDLE_FULL_BG" ]]; then
                echo "  ✓ Background model: $BUNDLE_FULL_BG"
            fi
            if [[ -n "$BUNDLE_FULL_EMBED" ]]; then
                echo "  ✓ Semantic search:  $BUNDLE_FULL_EMBED"
            fi
            ;;

        2)
            # Minimal bundle
            echo ""
            echo "  Downloading $BUNDLE_MIN_MAIN..."
            ollama pull "$BUNDLE_MIN_MAIN"

            export OLLAMA_MODEL="$BUNDLE_MIN_MAIN"

            # If it's a small model, it doubles as lightweight
            for entry in "${ALL_MODELS[@]}"; do
                local name=$(get_model_field "$entry" 1)
                local ram=$(get_model_field "$entry" 3)
                if [[ "$name" == "$BUNDLE_MIN_MAIN" && $ram -le 1000 ]]; then
                    export FAMILIAR_LIGHTWEIGHT_MODEL="$BUNDLE_MIN_MAIN"
                fi
            done

            MODELS_INSTALLED=true
            echo ""
            echo "  ✓ Model: $BUNDLE_MIN_MAIN"
            ;;

        3)
            # Custom multi-select
            run_custom_select
            ;;

        4)
            # Skip
            echo ""
            echo "  No models downloaded. Set an API key and re-run:"
            echo ""
            echo "    export ANTHROPIC_API_KEY='sk-ant-...'"
            echo "    ./run.sh"
            echo ""
            return 1
            ;;

        *)
            # Invalid — default to full
            echo "  Invalid choice, using Full Setup..."
            BUNDLE_CHOICE=1
            run_model_setup  # Recurse once
            return $?
            ;;
    esac

    # Persist to .env
    write_env ".env"
    return 0
}
