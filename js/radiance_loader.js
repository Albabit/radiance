/**
 * Radiance Unified Loader — Widget Management
 *
 * NOTE: This is a newly created file designed to assign default widget values 
 * specifically for LTX 2.3 presets at the moment. It establishes a scalable 
 * architecture and can easily be expanded in the future to support other models/presets.
 *
 * ALBABIT-FIX: Automatically populates loader widgets based on preset selection.
 * Aligned with the standard Radiance pattern using beforeRegisterNodeDef
 * to avoid fragile global nodeCreated race conditions.
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Radiance.UnifiedLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceUnifiedLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const presetW = this.widgets?.find(w => w.name === "preset");
            if (!presetW) return;

            const node = this;
            const origPreset = presetW.callback;

            // ALBABIT-FIX: Cache widget names to speed up the update process
            const widgetNames = [
                "unet_name", "weight_dtype", "model_type", "vae_name", 
                "audio_vae_name", "upscale_model_name", "llm_encoder", 
                "text_projection", "auto_download"
            ];

            presetW.callback = function () {
                if (origPreset) origPreset.apply(this, arguments);

                const val = presetW.value;
                if (val === "Custom" || val === "None (Custom)") return;

                // Create a temporary map for this execution for O(1) lookup speed
                const currentWidgets = {};
                node.widgets.forEach(w => {
                    if (widgetNames.includes(w.name)) currentWidgets[w.name] = w;
                });

                const setW = (name, value) => {
                    const w = currentWidgets[name];
                    if (w && w.value !== value) {
                        w.value = value;
                    }
                };

                // ── LTX Video 2.3 Preset Automation ──
                if (val.includes("→ LTX Video 2.3")) {
                    const isLowVram = val.includes("Low VRAM");

                    // 1. Common configuration for all LTX 2.3 workflows
                    setW("model_type", "ltxav");
                    setW("weight_dtype", "default");
                    setW("upscale_model_name", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors");
                    setW("auto_download", false); // ALBABIT-FIX: Fixed boolean assignment

                    if (!isLowVram) {
                        // 2. High Quality / Standard Mode
                        setW("unet_name", "ltx-2.3-22b-dev.safetensors");
                        setW("vae_name", "LTX23_video_vae_bf16.safetensors");
                        setW("audio_vae_name", "LTX23_audio_vae_bf16.safetensors");
                        setW("llm_encoder", "gemma_3_12B_it.safetensors");
                        setW("text_projection", "ltx-2.3_text_projection_bf16.safetensors");
                    } else {
                        // 3. Performance / Low VRAM Mode
                        setW("unet_name", "ltx-2.3-22b-dev-fp8.safetensors");
                        setW("vae_name", "Baked VAE (from UNET)");
                        setW("audio_vae_name", "Baked VAE (from UNET)");
                        setW("llm_encoder", "gemma_3_12B_it_fp4_mixed.safetensors");
                        setW("text_projection", "None");
                    }
                    
                    console.log(`[Radiance Loader] Preset ${val} optimized and applied.`);
                    node.setDirtyCanvas(true, true);
                }
            };
        };
    }
});