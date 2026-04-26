import { app } from "../../../scripts/app.js";

/**
 * Radiance AI Upscale Widget Visibility
 * Hides SUPIR-specific widgets when a non-SUPIR model is selected.
 *
 * SUPIR-only widgets: supir_steps, sdxl_model_name, supir_prompt
 * These are shown only when model_name is one of the SUPIR model identifiers.
 *
 * Uses the same three-mechanism visibility pattern as radiance_io.js:
 *   1. widget.options.hidden      — Nodes 2.0 Vue reactive filter
 *   2. widget.hidden              — LiteGraph getLayoutWidgets() exclusion
 *   3. type="hidden"+computeSize  — physical height collapse (all widget types)
 *      + draw=()=>{}              — prevents text bleeding on STRING widgets
 *      + inputEl/element display  — hides DOM node for customtext widgets
 */

// Must match _SUPIR_MODELS in upscale.py.
const SUPIR_MODEL_NAMES = ["SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"];

// ALBABIT-FIX: Optimal tile defaults for SUPIR (SDXL-native 1024px) vs other models.
const SUPIR_TILE_SIZE    = 1024;
const SUPIR_TILE_OVERLAP = 128;

// ALBABIT-FIX: Default values for all SUPIR-only widgets — applied on every manual switch to SUPIR.
// Must match Python INPUT_TYPES defaults in upscale.py.
const SUPIR_WIDGET_DEFAULTS = {
	supir_steps:      45,
	supir_s_churn:    5,
	seed:             1234,
	supir_cfg_start:  4.0,
	supir_cfg_end:    4.0,
	color_fix_type:   "None",
	supir_scale_by:   1.0,
	supir_restore_cfg: -1.0,
};

// Copied verbatim from radiance_io.js — shared pattern across all Radiance JS extensions.
function setWidgetVisible(widget, visible, node) {
	if (!widget) return;

	if (!widget.options) widget.options = {};
	widget.options.hidden = !visible;

	widget.hidden = !visible;
	if (visible) {
		if (widget.type === "hidden") {
			widget.type = widget._origType || "text";
			if (widget._origComputeSize !== undefined) {
				widget.computeSize = widget._origComputeSize;
			} else {
				delete widget.computeSize;
			}
			delete widget._origComputeSize;
			if (widget._origDraw !== undefined) {
				widget.draw = widget._origDraw;
				delete widget._origDraw;
			} else {
				delete widget.draw;
			}
			if (widget.inputEl) widget.inputEl.style.display = "";
			if (widget.element)  widget.element.style.display  = "";
			if (widget._origComputedHeight !== undefined) {
				widget.computedHeight = widget._origComputedHeight;
			}
			delete widget._origComputedHeight;
		}
	} else {
		if (widget.type !== "hidden") {
			widget._origType        = widget.type;
			widget._origComputeSize = widget.computeSize;
			widget._origComputedHeight = widget.computedHeight;
			widget.type = "hidden";
			widget.computeSize = () => [0, -4];
			if (widget.draw) widget._origDraw = widget.draw;
			widget.draw = function() {};
			if (widget.inputEl) widget.inputEl.style.display = "none";
			if (widget.element)  widget.element.style.display  = "none";
			widget.computedHeight = 4;
		}
	}
	if (node?.widgets) node.widgets.splice(0, 0);
}

// ALBABIT-FIX: Specialized visibility toggle for FLOAT/number widgets in Nodes 2.0.
// The standard setWidgetVisible sets type="hidden" to collapse height, but restoring a FLOAT
// widget's type is unreliable in Vue 3 (type string varies by ComfyUI version/renderer and
// the || "text" fallback may produce a wrong type → empty space after restore).
// This version avoids type mangling entirely: widget.options.hidden + widget.hidden handle
// Vue/LiteGraph layout exclusion; computeSize override handles physical height collapse.
function setFloatWidgetVisible(widget, visible, node) {
	if (!widget) return;

	if (!widget.options) widget.options = {};
	widget.options.hidden = !visible;
	widget.hidden         = !visible;

	if (visible) {
		if (widget._origComputeSize !== undefined) {
			widget.computeSize = widget._origComputeSize;
		} else {
			delete widget.computeSize;
		}
		delete widget._origComputeSize;
		if (widget._origComputedHeight !== undefined) {
			widget.computedHeight = widget._origComputedHeight;
		}
		delete widget._origComputedHeight;
		if (widget.inputEl) widget.inputEl.style.display = "";
		if (widget.element)  widget.element.style.display  = "";
	} else {
		widget._origComputeSize    = widget.computeSize;
		widget._origComputedHeight = widget.computedHeight;
		widget.computeSize    = () => [0, -4];
		widget.computedHeight = 4;
		if (widget.inputEl) widget.inputEl.style.display = "none";
		if (widget.element)  widget.element.style.display  = "none";
	}
	if (node?.widgets) node.widgets.splice(0, 0);
}

function refreshNodeSize(node) {
	if (node.computeSize) {
		const sz = node.computeSize();
		// Reassign as new array so Vue 3 reactive proxy detects the change.
		// Direct mutation (node.size[1] = x) bypasses Vue's setter.
		node.size = [Math.max(node.size[0], sz[0]), sz[1]];
		app.graph.setDirtyCanvas(true, true);
	}
}

app.registerExtension({
	name: "Radiance.AIUpscale",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "RadianceAIUpscale") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			const node = this;

			const modelWidget       = this.widgets.find(w => w.name === "model_name");
			const supirStepsWidget  = this.widgets.find(w => w.name === "supir_steps");
			const supirSChurnWidget = this.widgets.find(w => w.name === "supir_s_churn");
			const supirSeedWidget        = this.widgets.find(w => w.name === "seed");
			// ALBABIT-FIX: ComfyUI auto-appends a "control_after_generate" companion widget after "seed".
			const supirSeedControlWidget = this.widgets.find(w => w.name === "control_after_generate");
			const supirCfgStartWidget = this.widgets.find(w => w.name === "supir_cfg_start");
			const supirCfgEndWidget   = this.widgets.find(w => w.name === "supir_cfg_end");
			const colorFixWidget    = this.widgets.find(w => w.name === "color_fix_type");
			const scaleByWidget      = this.widgets.find(w => w.name === "supir_scale_by");
			const restoreCfgWidget   = this.widgets.find(w => w.name === "supir_restore_cfg");
			const sdxlModelWidget   = this.widgets.find(w => w.name === "sdxl_model_name");
			const supirPromptWidget = this.widgets.find(w => w.name === "supir_prompt");
			const scaleFactorWidget = this.widgets.find(w => w.name === "scale_factor");
			const tileSizeWidget    = this.widgets.find(w => w.name === "tile_size");
			const tileOverlapWidget = this.widgets.find(w => w.name === "tile_overlap");

			// ALBABIT-FIX: Apply SUPIR-optimal tile defaults on model switch; restore on exit.
			let _prevTileSize    = null;
			let _prevTileOverlap = null;
			let _lastWasSupir    = null;

			// ALBABIT-FIX: resetDefaults=true only on manual model switch (modelWidget.callback).
			// Timeouts and onConfigure pass false to preserve saved workflow values on load.
			const updateSupirWidgets = (resetDefaults = false) => {
				const isSupir = modelWidget
					? SUPIR_MODEL_NAMES.includes(modelWidget.value)
					: false;

				setWidgetVisible(supirStepsWidget,    isSupir, node);
				setWidgetVisible(supirSChurnWidget,   isSupir, node);
				setWidgetVisible(supirSeedWidget,        isSupir, node);
				setWidgetVisible(supirSeedControlWidget, isSupir, node);
				setWidgetVisible(supirCfgStartWidget, isSupir, node);
				setWidgetVisible(supirCfgEndWidget,   isSupir, node);
				setWidgetVisible(colorFixWidget,      isSupir, node);
				setWidgetVisible(scaleByWidget,       isSupir, node);
				setWidgetVisible(restoreCfgWidget,    isSupir, node);
				setWidgetVisible(sdxlModelWidget,     isSupir, node);
				setWidgetVisible(supirPromptWidget,   isSupir, node);
				// ALBABIT-FIX: scale_factor is feedforward-only; SUPIR uses supir_scale_by instead.
				// Uses setFloatWidgetVisible (no type mangling) to avoid empty-space on restore.
				setFloatWidgetVisible(scaleFactorWidget, !isSupir, node);

				if (resetDefaults) {
					if (isSupir) {
						// Save non-SUPIR tile values before first SUPIR entry in this session.
						if (_lastWasSupir !== true) {
							if (tileSizeWidget)    _prevTileSize    = tileSizeWidget.value;
							if (tileOverlapWidget) _prevTileOverlap = tileOverlapWidget.value;
						}
						// Always restore all SUPIR defaults on every manual switch to SUPIR.
						if (tileSizeWidget)      tileSizeWidget.value      = SUPIR_TILE_SIZE;
						if (tileOverlapWidget)   tileOverlapWidget.value   = SUPIR_TILE_OVERLAP;
						if (supirStepsWidget)    supirStepsWidget.value    = SUPIR_WIDGET_DEFAULTS.supir_steps;
						if (supirSChurnWidget)   supirSChurnWidget.value   = SUPIR_WIDGET_DEFAULTS.supir_s_churn;
						if (supirSeedWidget)     supirSeedWidget.value     = SUPIR_WIDGET_DEFAULTS.seed;
						if (supirCfgStartWidget) supirCfgStartWidget.value = SUPIR_WIDGET_DEFAULTS.supir_cfg_start;
						if (supirCfgEndWidget)   supirCfgEndWidget.value   = SUPIR_WIDGET_DEFAULTS.supir_cfg_end;
						if (colorFixWidget)      colorFixWidget.value      = SUPIR_WIDGET_DEFAULTS.color_fix_type;
						if (scaleByWidget)       scaleByWidget.value       = SUPIR_WIDGET_DEFAULTS.supir_scale_by;
						if (restoreCfgWidget)    restoreCfgWidget.value    = SUPIR_WIDGET_DEFAULTS.supir_restore_cfg;
						if (supirSeedControlWidget && supirSeedControlWidget.value === "randomize")
							supirSeedControlWidget.value = "fixed";
					} else if (_lastWasSupir === true) {
						// Restore non-SUPIR tile values when leaving SUPIR.
						if (tileSizeWidget    && _prevTileSize    !== null) tileSizeWidget.value    = _prevTileSize;
						if (tileOverlapWidget && _prevTileOverlap !== null) tileOverlapWidget.value = _prevTileOverlap;
						_prevTileSize    = null;
						_prevTileOverlap = null;
					}
				}
				_lastWasSupir = isSupir;

				refreshNodeSize(node);
			};

			if (modelWidget) {
				const origCb = modelWidget.callback;
				modelWidget.callback = function () {
					if (origCb) origCb.apply(this, arguments);
					// ALBABIT-FIX: resetDefaults=true — manual switch always restores all SUPIR defaults.
					updateSupirWidgets(true);
				};
			}

			// Pass 1 (100ms): hide widgets before Vue's first render.
			// Pass 2 (500ms): force correct height after Vue's initial layout,
			// with polling retry in case Vue defers layout further.
			setTimeout(updateSupirWidgets, 100);
			// ALBABIT-FIX: Store fn for onConfigure / loadedGraphNode / afterConfigureGraph.
			node._radianceUpdateVisibility = updateSupirWidgets;
			// ALBABIT-FIX: See radiance_io.js for rationale.
			const origUpscaleConfigure = node.onConfigure;
			node.onConfigure = function(info) {
				if (origUpscaleConfigure) origUpscaleConfigure.call(this, info);
				setTimeout(updateSupirWidgets, 10);
			};
			setTimeout(() => {
				refreshNodeSize(node);
				let retries = 0;
				const poll = setInterval(() => {
					if (!node.computeSize) { clearInterval(poll); return; }
					const target = node.computeSize()[1];
					if (Math.abs(node.size[1] - target) > 4) {
						node.size = [node.size[0], target];
						app.graph.setDirtyCanvas(true, true);
					}
					if (++retries >= 5) clearInterval(poll);
				}, 200);
			}, 500);

			return r;
		};
	},
	// ALBABIT-FIX: loadedGraphNode / afterConfigureGraph — see radiance_io.js for rationale.
	loadedGraphNode(node) {
		if (node._radianceUpdateVisibility) {
			setTimeout(node._radianceUpdateVisibility, 100);
		}
	},
	async afterConfigureGraph() {
		for (const node of app.graph._nodes) {
			if (node._radianceUpdateVisibility) {
				node._radianceUpdateVisibility();
			}
		}
	},
});
