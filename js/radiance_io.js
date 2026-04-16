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
	],
	"Sequence": [
		"Image Sequence — EXR (32-bit)",
		"Image Sequence — Radiance HDR (.hdr)",
		"Image Sequence — PNG (16-bit)",
		"Image Sequence — PNG (8-bit)",
		"Image Sequence — JPEG",
		"GIF — Animated",
		"WEBP — Animated",
	],
	// ALBABIT-FIX: Single Image mode uses shorter labels — "Image Sequence —" prefix removed
	// since there is no sequence. Values match the WRITE_FORMATS single-image entries in nodes_io.py.
	"Single Image": [
		"EXR (32-bit)",
		"Radiance HDR (.hdr)",
		"PNG (16-bit)",
		"PNG (8-bit)",
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
function setWidgetVisible(widget, visible) {
	if (!widget) return;
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
		}
	} else {
		if (widget.type !== "hidden") {
			widget._origType       = widget.type;
			widget._origComputeSize = widget.computeSize;
			widget.type = "hidden";
			widget.computeSize = () => [0, -4];
			// Mute draw to prevent text bleeding on hidden STRING widgets
			if (widget.draw) widget._origDraw = widget.draw;
			widget.draw = function() {};
			// Hide DOM node for STRING/customtext widgets
			if (widget.inputEl) widget.inputEl.style.display = "none";
			if (widget.element)  widget.element.style.display  = "none";
		}
	}
}

function refreshNodeSize(node) {
	if (node.computeSize) {
		const sz = node.computeSize();
		// ALBABIT-FIX: Force exact height (grow AND shrink) so the node resizes
		// correctly when widgets are shown or hidden dynamically.
		node.size[0] = Math.max(node.size[0], sz[0]);
		node.size[1] = sz[1];
		app.graph.setDirtyCanvas(true, true);
	}
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

		// 2. Digital Cinema Write — smart show/hide toggles
		// FIX 1: was "◎ RadianceDigitalCinemaWrite"
		if (nodeData.name === "RadianceDigitalCinemaWrite") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				const node = this;

				const writeModeWidget   = this.widgets.find(w => w.name === "write_mode");
				const formatWidget      = this.widgets.find(w => w.name === "output_format");
				const fpsWidget         = this.widgets.find(w => w.name === "fps");
				const startFrameWidget  = this.widgets.find(w => w.name === "start_frame");
				const bitDepthWidget    = this.widgets.find(w => w.name === "bit_depth");
				const compressionWidget = this.widgets.find(w => w.name === "compression");
				const alphaModeWidget   = this.widgets.find(w => w.name === "alpha_mode");
				const metadataWidget    = this.widgets.find(w => w.name === "custom_metadata");
				// ALBABIT-FIX: Audio export widgets — suffix only relevant when a format is chosen.
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

					const isVideo    = mode === "Video";
					const isSequence = mode === "Sequence";
					const isSingle   = mode === "Single Image";
					const isSeqLike  = isSequence || isSingle;

					// FPS only relevant for video
					setWidgetVisible(fpsWidget, isVideo);

					// Start frame only relevant for sequences
					setWidgetVisible(startFrameWidget, isSequence);

					// EXR/PNG-specific controls only in sequence/single modes
					setWidgetVisible(bitDepthWidget,    isSeqLike && (is_exr || is_png));
					setWidgetVisible(compressionWidget, isSeqLike && is_exr);
					setWidgetVisible(alphaModeWidget,   isSeqLike);
					setWidgetVisible(metadataWidget,    isSeqLike);

					// ALBABIT-FIX: Audio suffix is only relevant when an export format is selected.
					// audio_export itself is always visible (can also save audio alongside a video).
					const audioExportValue = audioExportWidget ? audioExportWidget.value : "None";
					setWidgetVisible(audioSuffixWidget, audioExportValue !== "None");

					refreshNodeSize(node);
				};

				if (writeModeWidget)   writeModeWidget.callback   = updateWidgets;
				if (formatWidget)      formatWidget.callback      = updateWidgets;
				// ALBABIT-FIX: Recompute suffix visibility when write_audio format changes.
				if (audioExportWidget) audioExportWidget.callback = updateWidgets;

				// ALBABIT-FIX: 100ms delay (was 20ms) to ensure ComfyUI has fully initialised
				// all widgets — especially multiline STRING textareas whose computeSize
				// may not be set yet at 20ms, causing a stale save and a 20px-high restore.
				setTimeout(updateWidgets, 100);

				return r;
			};
		}
	}
});
