import { app } from "../../../../scripts/app.js";

const lastDimensionsByNode = new WeakMap();

function parseDimensions(str) {
  if (!str || typeof str !== "string") return [];
  const re = /(\d+)\s*x\s*(\d+)/i;
  return str
    .split(/\s*,\s*/)
    .map((dim) => {
      const m = re.exec(dim.trim());
      return m ? `${m[1]}x${m[2]}` : null;
    })
    .filter(Boolean);
}

function updatePresetOptions(node) {
  const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
  const presetWidget = node.widgets?.find((w) => w.name === "preset");
  if (!dimensionsWidget || !presetWidget || presetWidget.type !== "combo") return;
  const text = dimensionsWidget.value ?? "";
  const options = [...new Set(parseDimensions(text))];
  if (options.length === 0) return;
  presetWidget.options.values = options;
  if (!options.includes(presetWidget.value)) presetWidget.value = options[0];
  node.setDirtyCanvas(true, true);
}

function pollDimensionsNodes() {
  const graph = app.graph;
  if (!graph?._nodes) return;
  graph._nodes.forEach((node) => {
    if (node.comfyClass !== "DimensionsPresetPicker") return;
    const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
    if (!dimensionsWidget) return;
    const current = dimensionsWidget.value ?? "";
    if (current !== lastDimensionsByNode.get(node)) {
      lastDimensionsByNode.set(node, current);
      updatePresetOptions(node);
    }
  });
}

let pollInterval = null;

function ensurePolling() {
  if (pollInterval != null) return;
  pollInterval = setInterval(pollDimensionsNodes, 350);
}

app.registerExtension({
  name: "basenc.DimensionsPresetPicker",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "DimensionsPresetPicker") return;
    const dimensionsInput = nodeData.input?.required?.dimensions;
    const presetInput = nodeData.input?.required?.preset;
    if (presetInput && presetInput[0] === "COMBO") {
      presetInput[1] = presetInput[1] || {};
      const defaultDimensions =
        dimensionsInput?.[1]?.default ?? "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768";
      presetInput[1].values = parseDimensions(defaultDimensions);
    }
  },

  nodeCreated(node) {
    if (node.comfyClass !== "DimensionsPresetPicker") return;
    const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
    if (dimensionsWidget) lastDimensionsByNode.set(node, dimensionsWidget.value ?? "");
    updatePresetOptions(node);
    ensurePolling();
  },
});
