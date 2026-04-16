import { app } from "../../../scripts/app.js";
// FIX 5: was "../../scripts/app.js" — extensions in custom_nodes/Radiance/web/
// need three ../ to reach ComfyUI's scripts/ directory.

/**
 * Radiance Resolution — Widget Management (v2.3)
 *
 * FIX 5: Import path corrected (see above).
 * FIX 6: Switched from nodeCreated hook + nested setTimeout to beforeRegisterNodeDef
 * which is the standard Radiance pattern. nodeCreated fires for every node
 * in the graph including unrelated ones; beforeRegisterNodeDef targets
 * exactly our node type before registration, avoiding the fragile 10ms
 * race condition on widget availability.
 *
 * FEATURE: Shows/hides mp_target and mp_aspect_ratio widgets based on whether
 * mp_target > 0, alongside the existing video/batch toggle.
 *
 * ALBABIT-FIX: Fixed ReferenceError 'r' by properly defining it from the original callback.
 * ALBABIT-FIX: Auto-fills W/H inputs when selecting a preset.
 * ALBABIT-FIX: Toggles crop_to_broadcast_resolution based on enable_video.
 * ALBABIT-FIX: Greys out mp_aspect_ratio when mp_target is 0.
 */

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "INT";
            widget.computeSize = widget._origComputeSize || (() => [200, 20]);
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
    }
}

function refreshNodeSize(node) {
    // Single deferred resize — avoids double-setTimeout pattern
    if (node.computeSize) {
        const sz = node.computeSize();
        if (node.size[0] < sz[0]) node.size[0] = sz[0];
        if (node.size[1] < sz[1]) node.size[1] = sz[1];
        app.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "radiance.resolution",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RadianceResolution") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                // ALBABIT-FIX: Properly define 'r' to fix the ReferenceError crash
                const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

                const enableVideoW = this.widgets?.find(w => w.name === "enable_video");
                const videoFramesW = this.widgets?.find(w => w.name === "video_frames");
                const frameRateW   = this.widgets?.find(w => w.name === "frame_rate");
                const batchSizeW   = this.widgets?.find(w => w.name === "batch_size");
                const mpTargetW    = this.widgets?.find(w => w.name === "mp_target");
                const mpAspectW    = this.widgets?.find(w => w.name === "mp_aspect_ratio");
                
                // ALBABIT-FIX: New widgets for time computation
                const frameModeW   = this.widgets?.find(w => w.name === "frame_computation");
                const durSecW      = this.widgets?.find(w => w.name === "duration_seconds");
                // ALBABIT-FIX: New widgets required for UI logic
                const presetW      = this.widgets?.find(w => w.name === "preset");
                const widthW       = this.widgets?.find(w => w.name === "width");
                const heightW      = this.widgets?.find(w => w.name === "height");
                const cropBroadcastW = this.widgets?.find(w => w.name === "crop_to_broadcast_resolution");

                const toggleFields = () => {
                    if (!enableVideoW) return;

                    const isVideo = enableVideoW.value === true
                        || enableVideoW.value === "true"
                        || enableVideoW.value === "True";

                    // ALBABIT-FIX: Toggle between manual frames and auto seconds
                    const isAutoSec = frameModeW && (frameModeW.value === "Auto (Seconds)");
                    
                    setWidgetVisible(frameModeW, isVideo);
                    setWidgetVisible(videoFramesW, isVideo && !isAutoSec);
                    setWidgetVisible(durSecW, isVideo && isAutoSec);
                    setWidgetVisible(frameRateW,   isVideo);
                    setWidgetVisible(batchSizeW,   !isVideo);
                    
                    // ALBABIT-FIX: Show crop_to_broadcast_resolution only when video is enabled
                    setWidgetVisible(cropBroadcastW, isVideo);

                    // FEATURE: mp_aspect_ratio only visible when mp_target > 0
                    if (mpTargetW && mpAspectW) {
                        const mpActive = parseFloat(mpTargetW.value) > 0;
                        setWidgetVisible(mpAspectW, mpActive);
                        
                        // ALBABIT-FIX: Grey out the aspect ratio widget when inactive
                        mpAspectW.disabled = !mpActive;
                        if (mpAspectW.inputEl) {
                            mpAspectW.inputEl.disabled = !mpActive;
                            mpAspectW.inputEl.style.opacity = mpActive ? "1.0" : "0.4";
                        }
                    }

                    refreshNodeSize(this);
                };

                // Wire callbacks
                if (enableVideoW) {
                    const orig = enableVideoW.callback;
                    enableVideoW.callback = function () {
                        if (orig) orig.apply(this, arguments);
                        toggleFields();
                    };
                }

                // ALBABIT-FIX: Wire callback for frame computation mode
                if (frameModeW) {
                    const orig = frameModeW.callback;
                    frameModeW.callback = function () {
                        if (orig) orig.apply(this, arguments);
                        toggleFields();
                    };
                }

                if (mpTargetW) {
                    const orig = mpTargetW.callback;
                    mpTargetW.callback = function () {
                        if (orig) orig.apply(this, arguments);
                        toggleFields();
                    };
                }

                // ALBABIT-FIX: Auto-update width/height, enable_video, and model_type on preset change
                if (presetW && widthW && heightW) {
                    const origPreset = presetW.callback;
                    const node = this; // ALBABIT-FIX: Save the Node reference here to avoid 'this' context issues

                    presetW.callback = function () {
                        // ALBABIT-FIX: Here 'this' refers to the Widget, keeping ComfyUI's native callback happy!
                        if (origPreset) origPreset.apply(this, arguments);
                        
                        if (presetW.value !== "Custom" && presetW.value !== "None (Custom)") {
                            // Extract dimensions
                            const match = presetW.value.match(/\((\d+)[x×](\d+)\)/);
                            if (match) {
                                widthW.value = parseInt(match[1], 10);
                                heightW.value = parseInt(match[2], 10);
                                if (widthW.inputEl) widthW.inputEl.value = widthW.value;
                                if (heightW.inputEl) heightW.inputEl.value = heightW.value;
                            }

                            // Auto-toggle Video Mode for known video presets
                            if (enableVideoW) {
                                const isVideoPreset = presetW.value.includes("LTX") || presetW.value.includes("WAN") || presetW.value.includes("HunyuanVideo");
                                if (enableVideoW.value !== isVideoPreset) {
                                    enableVideoW.value = isVideoPreset;
                                    if (enableVideoW.callback) enableVideoW.callback.call(enableVideoW, isVideoPreset);
                                }
                            }

                            // Auto-set Model Type for LTX
                            // ALBABIT-FIX: We use 'node.widgets' instead of 'this.widgets'
                            const modelTypeW = node.widgets?.find(w => w.name === "model_type");
                            const latentChannelsW = node.widgets?.find(w => w.name === "latent_channels");
                            // ALBABIT-FIX: Find the scale_factor widget
                            const scaleFactorW = node.widgets?.find(w => w.name === "scale_factor");

                            // ALBABIT-FIX: Auto-adjust scale_factor based on LTX workflow requirements
                            if (scaleFactorW) {
                                if (presetW.value.includes("4x latent upscale wf")) {
                                    scaleFactorW.value = 0.25; // Divide by 4
                                } else if (presetW.value.includes("LTX")) {
                                    scaleFactorW.value = 0.5;  // Standard divide by 2
                                } else {
                                    scaleFactorW.value = 1.0;  // Reset to 1 for other models (Flux, etc.)
                                }
                            }

                            if (modelTypeW) {
                                if (presetW.value.includes("LTX")) {
                                    modelTypeW.value = "LTXV (128ch)";
                                    if (latentChannelsW) latentChannelsW.value = 128;
                                } else if (presetW.value.includes("SDXL") || presetW.value.includes("SD 1.5")) {
                                    modelTypeW.value = "SDXL / SD 1.5 (4ch)";
                                    if (latentChannelsW) latentChannelsW.value = 0;
                                } else {
                                    modelTypeW.value = "Auto (Flux 16ch / LTXV)";
                                    if (latentChannelsW) latentChannelsW.value = 0;
                                }
                            }
                        }
                    };
                }

                // Initial state — widgets exist here (no setTimeout needed)
                toggleFields();
                
                // ALBABIT-FIX: Ensure default initialization and persistent visibility on tab switch
                setTimeout(() => {
                    // Sync Preset Dimensions
                    if (presetW && presetW.value !== "Custom" && presetW.value !== "None (Custom)" && widthW && heightW) {
                        const match = presetW.value.match(/\((\d+)[x×](\d+)\)/);
                        if (match && widthW.value === 1024 && heightW.value === 1024) {
                            widthW.value = parseInt(match[1], 10);
                            heightW.value = parseInt(match[2], 10);
                        }
                    }
                    // Force UI visibility sync based on current toggle states
                    toggleFields();
                }, 100);

                return r;
            };
        }
    }
});