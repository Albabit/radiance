import { app } from "../../../scripts/app.js";

/**
 * Radiance Universal I/O Widget Management (v2.3)
 * Handles dynamic visibility for Digital Cinema Read and Write nodes.
 *
 * Fixes applied:
 *  FIX 1: Node names updated to match nodes_io.py after ◎ was removed from
 *          NODE_CLASS_MAPPINGS keys. Extension was completely dead before.
 *  FIX 2: Full widget visibility via setWidgetVisible(): type="hidden"+computeSize=[0,-4]
 *          collapses height; draw()={} prevents text bleeding on STRING widgets;
 *          inputEl/element display:none hides the DOM node for customtext widgets.
 *  FIX 3: Removed origType forEach pre-init; _origType/_origComputeSize/_origDraw are
 *          saved lazily inside setWidgetVisible() on first hide.
 *  FIX 4: Read node calls sourceWidget.callback() at init so label is correct
 *          when a saved workflow is restored.
 *  FIX 5: Read node isVideo check extended to include .avi .mkv .webm,
 *          matching the Python reader's accepted extensions exactly.
 */

// FIX 5: Single source-of-truth for video extensions — mirrors Python read().
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm"];

// ALBABIT-FIX: Output format options grouped by write_mode — mirrors WRITE_FORMATS in nodes_io.py.
// Filtering the combo dynamically prevents invalid mode/format combinations.
// GIF/WEBP are no longer in Video — animated variants belong to Sequence,
// static variants to Single Image. Video mode is strictly for cinema codecs.
const FORMAT_GROUPS = {
	"Video": [
		"Video — MP4 (H.264)",
		"Video — MP4 (H.265 10-bit)",
		"Video — MOV (ProRes 422 HQ)",
		"Video — MOV (ProRes 4444)",
		"Video — MOV (ProRes 4444 XQ)",
		"Video — MOV (ProRes 4444 HDR Log)",
		"Video — MOV (DNxHR HQ)",
		"Video — MOV (DNxHR 444)",
	],
	"Sequence": [
		"Image Sequence — EXR (32-bit)",
		"Image Sequence — Radiance HDR (.hdr)",
		"Image Sequence — PNG (16-bit)",
		"Image Sequence — PNG (8-bit)",
		"Image Sequence — TIFF (32-bit Float)",
		"Image Sequence — TIFF (16-bit)",
		"Image Sequence — JPEG",
		"GIF — Animated",
		"WEBP — Animated",
	],
	// Single Image mode uses shorter labels — "Image Sequence —" prefix removed.
	// Values match the WRITE_FORMATS single-image entries in nodes_io.py.
	"Single Image": [
		"EXR (32-bit)",
		"Radiance HDR (.hdr)",
		"PNG (16-bit)",
		"PNG (8-bit)",
		"TIFF (32-bit Float)",
		"TIFF (16-bit)",
		"JPEG",
		"GIF",
		"WEBP",
	],
};

// ALBABIT-FIX: Full widget visibility — combines radiance_resolution.js + radiance_sampler.js patterns.
// Three mechanisms are required to fully hide a widget:
//   1. type="hidden" + computeSize=[0,-4]  — collapses height for all widget types.
//   2. widget.draw = () => {}              — prevents text/label bleeding for STRING widgets
//                                            that still call their draw() even when "hidden".
//   3. inputEl/element display:none        — hides the actual DOM node for customtext (STRING)
//                                            widgets, which are absolutely-positioned over the canvas.
// ALBABIT-FIX: Added node parameter to support Nodes 2.0 Vue reactive widget hiding.
// Nodes 2.0 uses widget.options.hidden to filter widgets from Vue rendering
// (confirmed in ComfyUI frontend source: t.filter(e=>!(e.options?.hidden||...)))
function setWidgetVisible(widget, visible, node) {
	if (!widget) return;

	// ALBABIT-FIX: Nodes 2.0 primary mechanism — options.hidden filters widget from Vue render list
	if (!widget.options) widget.options = {};
	widget.options.hidden = !visible;

	// Classic LiteGraph canvas: widget.hidden drives getLayoutWidgets() exclusion
	widget.hidden = !visible;
	if (visible) {
		if (widget.type === "hidden") {
			widget.type = widget._origType || "text";
			// ALBABIT-FIX: Use delete rather than restoring a saved computeSize value.
			// If computeSize was not yet initialised by ComfyUI when we first hid the widget,
			// _origComputeSize would be undefined and the fallback [200,20] gives a 20px textarea
			// (the "tiny scrollbar" bug). Deleting the override lets LiteGraph fall back to
			// the prototype computeSize, which always returns the correct height.
			delete widget.computeSize;
			delete widget._origComputeSize;
			// Restore draw function (saved on hide)
			if (widget._origDraw !== undefined) {
				widget.draw = widget._origDraw;
				delete widget._origDraw;
			} else {
				delete widget.draw;
			}
			// Restore DOM visibility for STRING/customtext widgets
			if (widget.inputEl) widget.inputEl.style.display = "";
			if (widget.element)  widget.element.style.display  = "";
			// ALBABIT-FIX: Restore saved computedHeight for Nodes 2.0 Vue layout.
			// If the widget was hidden before Vue's first layout pass, fall back to 32 (standard row height).
			if (widget._origComputedHeight !== undefined) {
				widget.computedHeight = widget._origComputedHeight;
				delete widget._origComputedHeight;
			} else {
				widget.computedHeight = 32;
			}
		}
	} else {
		if (widget.type !== "hidden") {
			widget._origType       = widget.type;
			widget._origComputeSize = widget.computeSize;
			// ALBABIT-FIX: Save computedHeight so the show path can restore it exactly
			widget._origComputedHeight = widget.computedHeight;
			widget.type = "hidden";
			widget.computeSize = () => [0, -4];
			// Mute draw to prevent text bleeding on hidden STRING widgets
			if (widget.draw) widget._origDraw = widget.draw;
			widget.draw = function() {};
			// Hide DOM node for STRING/customtext widgets
			if (widget.inputEl) widget.inputEl.style.display = "none";
			if (widget.element)  widget.element.style.display  = "none";
			// ALBABIT-FIX: Nodes 2.0 — set computedHeight=4 so Vue CSS height becomes 0px (collapses widget)
			widget.computedHeight = 4;
		}
	}
	// ALBABIT-FIX: Always splice to trigger Vue reactive proxy re-evaluation of options.hidden,
	// even when widget.type was never "hidden" (e.g. showing a widget on fresh node load)
	if (node?.widgets) node.widgets.splice(0, 0);
}

function refreshNodeSize(node) {
	if (node.computeSize) {
		const sz = node.computeSize();
		const newW = Math.max(node.size[0], sz[0]);
		const newH = sz[1];
		// ALBABIT-FIX: Reassign as new array so Vue 3 reactive proxy detects the change.
		// Direct mutation (node.size[1] = x) bypasses Vue's setter and leaves blank gaps
		// when widgets are hidden and the saved node.size is larger than the collapsed sum.
		node.size = [newW, newH];
		app.graph.setDirtyCanvas(true, true);
	}
}

function refreshNodeSizeWithRetry(node) {
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
}

app.registerExtension({
	name: "Radiance.IO",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		// 1. Digital Cinema Read — label intelligence
		// FIX 1: was "◎ RadianceDigitalCinemaRead" — ◎ removed from Python mapping key.
		if (nodeData.name === "RadianceDigitalCinemaRead") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const sourceWidget = this.widgets.find(w => w.name === "source_path");

				if (sourceWidget) {
					// FIX 5: use shared VIDEO_EXTENSIONS list
					const updateLabel = () => {
						const val = (sourceWidget.value || "").toLowerCase();
						const isVideo = VIDEO_EXTENSIONS.some(ext => val.endsWith(ext));
						sourceWidget.label = isVideo ? "SOURCE (VIDEO)" : "SOURCE (SEQUENCE/IMAGE)";
					};

					sourceWidget.callback = updateLabel;

					// FIX 4: run immediately so restored workflows show the right label
					setTimeout(updateLabel, 20);
				}

				return r;
			};
		}

		// 2. Digital Cinema Read — widget visibility by read_mode
		if (nodeData.name === "RadianceDigitalCinemaRead") {
			const onNodeCreatedRead = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreatedRead ? onNodeCreatedRead.apply(this, arguments) : undefined;
				const node = this;

				const readModeWidget   = this.widgets.find(w => w.name === "read_mode");
				const startFrameWidget = this.widgets.find(w => w.name === "start_frame");
				const frameLimitWidget = this.widgets.find(w => w.name === "frame_limit");
				const frameNumberWidget = this.widgets.find(w => w.name === "frame_number");
				const fpsOverrideWidget = this.widgets.find(w => w.name === "fps_override");

				const updateReadWidgets = () => {
					const mode = readModeWidget ? readModeWidget.value : "Auto";
					const isSingle = mode === "Single Frame";

					setWidgetVisible(startFrameWidget,  !isSingle, node);
					setWidgetVisible(frameLimitWidget,  !isSingle, node);
					setWidgetVisible(frameNumberWidget,  isSingle, node);
					setWidgetVisible(fpsOverrideWidget, !isSingle, node);

					refreshNodeSize(node);
				};

				if (readModeWidget) {
					const origReadCb = readModeWidget.callback;
					readModeWidget.callback = function () {
						if (origReadCb) origReadCb.apply(this, arguments);
						updateReadWidgets();
					};
				}

				setTimeout(updateReadWidgets, 100);
				// ALBABIT-FIX: Store fn for loadedGraphNode / afterConfigureGraph.
				node._radianceUpdateVisibility = updateReadWidgets;
				// ALBABIT-FIX: onConfigure fires after widgets_values are restored from the
				// saved graph. setTimeout(10) lets Vue settle reactivity before we set
				// options.hidden, fixing the "widgets reappear on refresh" bug in Nodes 2.0.
				const origReadConfigure = node.onConfigure;
				node.onConfigure = function(info) {
					if (origReadConfigure) origReadConfigure.call(this, info);
					setTimeout(updateReadWidgets, 10);
				};
				setTimeout(() => refreshNodeSizeWithRetry(node), 500);

				return r;
			};
		}

		// 3. Digital Cinema Write — smart show/hide toggles
		// FIX 1: was "◎ RadianceDigitalCinemaWrite"
		if (nodeData.name === "RadianceDigitalCinemaWrite") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				const node = this;

				const writeModeWidget   = this.widgets.find(w => w.name === "write_mode");
				const formatWidget      = this.widgets.find(w => w.name === "output_format");
				const qualityWidget     = this.widgets.find(w => w.name === "quality");
				const fpsWidget         = this.widgets.find(w => w.name === "fps");
				const startFrameWidget  = this.widgets.find(w => w.name === "start_frame");
				const bitDepthWidget    = this.widgets.find(w => w.name === "bit_depth");
				const compressionWidget = this.widgets.find(w => w.name === "compression");
				const alphaModeWidget   = this.widgets.find(w => w.name === "alpha_mode");
				const metadataWidget    = this.widgets.find(w => w.name === "custom_metadata");
				// ALBABIT-FIX: Audio export widgets — suffix only relevant when a format is chosen.
				const framePaddingWidget = this.widgets.find(w => w.name === "frame_padding");
				const audioExportWidget = this.widgets.find(w => w.name === "write_external_audio_file");
				const audioSuffixWidget = this.widgets.find(w => w.name === "audio_filename_suffix");

				const updateWidgets = () => {
					const mode = writeModeWidget ? writeModeWidget.value : "Video";

					// ALBABIT-FIX: Filter output_format options to only show formats valid for
					// the current write_mode. If the current value is no longer in the valid
					// list, reset to the first option of the new group.
					if (formatWidget) {
						const validFormats = FORMAT_GROUPS[mode] || FORMAT_GROUPS["Video"];
						if (!validFormats.includes(formatWidget.value)) {
							formatWidget.value = validFormats[0];
						}
						formatWidget.options.values = validFormats;
					}

					const fmt    = formatWidget ? formatWidget.value : "";
					const is_exr = fmt.includes("EXR");
					const is_png = fmt.includes("PNG");
					const is_jpg = fmt.includes("JPEG");
					const is_webp = fmt.includes("WEBP");
					const is_crf = fmt.includes("H.264") || fmt.includes("H.265");

					const isVideo    = mode === "Video";
					const isSequence = mode === "Sequence";
					const isSingle   = mode === "Single Image";
					const isSeqLike  = isSequence || isSingle;

					// ALBABIT-FIX: Dynamic quality label reflects the active codec's quality scale.
					// ALBABIT-FIX: Hide quality when the selected format ignores it entirely
					// (EXR, PNG, ProRes, Radiance HDR). Only H.264/H.265, JPEG, and WEBP use it.
					const qualityUsed = is_crf || is_jpg || is_webp;
					if (qualityWidget) {
						qualityWidget.label = is_crf
							? "quality (CRF)"
							: (is_jpg || is_webp) ? "quality (0–100)"
							: "quality";
					}
					// ALBABIT-FIX: Pass node for Nodes 2.0 Vue reactive widget hiding
					setWidgetVisible(qualityWidget, qualityUsed, node);

					// FPS only relevant for video
					setWidgetVisible(fpsWidget, isVideo, node);

					// Start frame and frame_padding only relevant for sequences
					setWidgetVisible(startFrameWidget,   isSequence, node);
					// frame_padding controls zero-padding in filenames (e.g. 0001 vs 000001) —
					// meaningless for Single Image since only one file is written.
					setWidgetVisible(framePaddingWidget, isSequence, node);

					// ALBABIT-FIX: bit_depth is EXR-only (Half Float vs Full Float).
					// PNG encodes depth in the format name; HDR is always 32-bit RGBE — both ignore this param.
					setWidgetVisible(bitDepthWidget,    isSeqLike && is_exr, node);
					setWidgetVisible(compressionWidget, isSeqLike && is_exr, node);
					setWidgetVisible(alphaModeWidget,   isSeqLike, node);
					setWidgetVisible(metadataWidget,    isSeqLike, node);

					// ALBABIT-FIX: Audio suffix is only relevant when an export format is selected.
					// audio_export itself is always visible (can also save audio alongside a video).
					const audioExportValue = audioExportWidget ? audioExportWidget.value : "None";
					setWidgetVisible(audioSuffixWidget, audioExportValue !== "None", node);

					refreshNodeSize(node);
				};

				if (writeModeWidget)   writeModeWidget.callback   = updateWidgets;
				if (formatWidget)      formatWidget.callback      = updateWidgets;
				// ALBABIT-FIX: Recompute suffix visibility when write_audio format changes.
				if (audioExportWidget) audioExportWidget.callback = updateWidgets;

				// ALBABIT-FIX: 100ms delay to ensure ComfyUI has fully initialised all widgets
				// (especially multiline STRING textareas). Retry resize for Vue reactivity.
				setTimeout(updateWidgets, 100);
				// ALBABIT-FIX: Store fn for loadedGraphNode / afterConfigureGraph.
				node._radianceUpdateVisibility = updateWidgets;
				// ALBABIT-FIX: See Read node comment above — same rationale.
				const origWriteConfigure = node.onConfigure;
				node.onConfigure = function(info) {
					if (origWriteConfigure) origWriteConfigure.call(this, info);
					setTimeout(updateWidgets, 10);
				};
				setTimeout(() => refreshNodeSizeWithRetry(node), 500);

				return r;
			};
		}
	},
	// ALBABIT-FIX: loadedGraphNode fires after ALL nodes are configured and
	// widgets_values are fully restored — catches cases where onConfigure alone
	// fires too early for Vue to have mounted its reactive component.
	loadedGraphNode(node) {
		if (node._radianceUpdateVisibility) {
			setTimeout(node._radianceUpdateVisibility, 100);
		}
	},
	// ALBABIT-FIX: afterConfigureGraph is async — by the time it fires, Vue has
	// crossed its await boundary and node.widgets is a reactive proxy. A direct
	// call (no inner setTimeout) triggers splice(0,0) on the live reactive array,
	// forcing Vue to re-render with the correct options.hidden state.
	// This is the primary fix for saved-workflow restore in Nodes 2.0.
	async afterConfigureGraph() {
		for (const node of app.graph._nodes) {
			if (node._radianceUpdateVisibility) {
				node._radianceUpdateVisibility();
			}
		}
	},
});
