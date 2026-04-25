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

// Copied verbatim from radiance_io.js — shared pattern across all Radiance JS extensions.
function setWidgetVisible(widget, visible, node) {
	if (!widget) return;

	if (!widget.options) widget.options = {};
	widget.options.hidden = !visible;

	widget.hidden = !visible;
	if (visible) {
		if (widget.type === "hidden") {
			widget.type = widget._origType || "text";
			delete widget.computeSize;
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
				delete widget._origComputedHeight;
			} else {
				widget.computedHeight = 32;
			}
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
			const sdxlModelWidget   = this.widgets.find(w => w.name === "sdxl_model_name");
			const supirPromptWidget = this.widgets.find(w => w.name === "supir_prompt");

			const updateSupirWidgets = () => {
				const isSupir = modelWidget
					? SUPIR_MODEL_NAMES.includes(modelWidget.value)
					: false;

				setWidgetVisible(supirStepsWidget,  isSupir, node);
				setWidgetVisible(sdxlModelWidget,   isSupir, node);
				setWidgetVisible(supirPromptWidget, isSupir, node);

				refreshNodeSize(node);
			};

			if (modelWidget) {
				const origCb = modelWidget.callback;
				modelWidget.callback = function () {
					if (origCb) origCb.apply(this, arguments);
					updateSupirWidgets();
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
