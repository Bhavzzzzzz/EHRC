#!/bin/bash

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
NAS_BASE="/run/user/1017/gvfs/smb-share:server=172.16.202.70,share=home/biomarkers"
NAS_RAW_DIR="${NAS_BASE}/raw_data"
NAS_UNNORM_ARCHIVE_DIR="${NAS_BASE}/archived_unnormalized_patches"
NAS_ARCHIVE_DIR="${NAS_BASE}/archived_patches"

LOCAL_WORKSPACE="${HOME}/local_workspace"
LOCAL_RAW="${LOCAL_WORKSPACE}/raw_data"
LOCAL_UNNORM="${LOCAL_WORKSPACE}/unnormalized_patches"
LOCAL_NORM="${LOCAL_WORKSPACE}/normalized_patches"
LOCAL_STAGING="${LOCAL_WORKSPACE}/staging" 

# ==========================================
# SAFETY CATCH: Check if NAS is actually mounted
# ==========================================
if [ ! -d "$NAS_RAW_DIR" ]; then
    echo "🚨 ERROR: The NAS is not mounted or the path is broken!"
    exit 1
fi

mkdir -p "$NAS_UNNORM_ARCHIVE_DIR" "$NAS_ARCHIVE_DIR" "$LOCAL_STAGING"

# ==========================================
# 2. THE PROCESSING LOOP
# ==========================================
for vsi_path in "$NAS_RAW_DIR"/*/*.vsi; do
    
    [ -e "$vsi_path" ] || continue

    slide_name=$(basename "$vsi_path" .vsi)
    vsi_dir=$(dirname "$vsi_path")
    
    if [ -f "$NAS_ARCHIVE_DIR/${slide_name}_normalized.tar" ]; then
        echo "⏩ Skipping ${slide_name} - patches already archived on NAS."
        continue
    fi

    # --- THE TRAFFIC LIGHT (Prevents NVMe overflow) ---
    # It counts the .tar files in the staging folder. 4 tar files = 2 slides in the queue.
    while [ $(find "$LOCAL_STAGING" -name "*.tar" 2>/dev/null | wc -l) -ge 4 ]; do
        echo "⏳ TRAFFIC JAM: Staging area is full. Waiting 60 seconds for network uploads to catch up..."
        sleep 60
    done

    echo "====================================================="
    echo "▶️ STARTING SLIDE: ${slide_name}"
    echo "====================================================="

    # --- Step A: Prepare fresh local workspace ---
    echo "🧹 Cleaning active workspace on NVMe..."
    rm -rf "${LOCAL_RAW:?}"/* "${LOCAL_UNNORM:?}"/* "${LOCAL_NORM:?}"/*
    mkdir -p "$LOCAL_RAW" "$LOCAL_UNNORM" "$LOCAL_NORM"

    # --- Step B: Pull data from NAS to Local NVMe ---
    echo "⬇️ Copying ${slide_name}.vsi to local drive..."
    cp "$vsi_path" "$LOCAL_RAW/"
    
    vsi_data_folder="${vsi_dir}/_${slide_name}_"
    if [ -d "$vsi_data_folder" ]; then
        echo "⬇️ Copying associated data folder _${slide_name}_ ..."
        cp -r "$vsi_data_folder" "$LOCAL_RAW/"
    fi

    # --- Step C: Run Phase 1 & 2 (Image Generation) ---
    echo "🧠 Running Phase 1: Patch Extraction..."
    python3 extract_patches.py
    
    echo "🎨 Running Phase 2: Stain Normalization..."
    python3 normalize_patches.py

    if [ ! -d "$LOCAL_UNNORM/${slide_name}" ] || [ -z "$(ls -A "$LOCAL_UNNORM/${slide_name}")" ]; then
        echo "⚠️ WARNING: No patches were extracted for ${slide_name} (likely blank glass)."
        echo "Moving to the next slide..."
        continue
    fi

    # --- Step D: ASYNCHRONOUS Archive and Push ---
    echo "📦 Handing off ${slide_name} to background uploader..."
    
    mv "$LOCAL_UNNORM/${slide_name}" "$LOCAL_STAGING/${slide_name}_unnorm"
    mv "$LOCAL_NORM/${slide_name}" "$LOCAL_STAGING/${slide_name}_norm"

    (
        cd "$LOCAL_STAGING" || exit
        
        tar -cf "${slide_name}_unnormalized.tar" "${slide_name}_unnorm/"
        tar -cf "${slide_name}_normalized.tar" "${slide_name}_norm/"

        # Fix for SMB Permission Error: Use 'cp' instead of 'mv'
        cp "${slide_name}_unnormalized.tar" "$NAS_UNNORM_ARCHIVE_DIR/"
        cp "${slide_name}_normalized.tar" "$NAS_ARCHIVE_DIR/"

        # Delete the local staging folders AND the local .tar files after copying
        rm -rf "${slide_name}_unnorm" "${slide_name}_norm" "${slide_name}_unnormalized.tar" "${slide_name}_normalized.tar"
        
        echo -e "\n✅ [BACKGROUND TASK COMPLETE]: ${slide_name} safely secured on NAS.\n"
    ) & 
    
    echo "⏩ Upload running in background. Moving to extract the next slide IMMEDIATELY!"
    echo "====================================================="
done

echo "🎉 ALL SLIDES QUEUED! Waiting for final background uploads to finish..."
wait 
echo "✅ PIPELINE 100% COMPLETE."
