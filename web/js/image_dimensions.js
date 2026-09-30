import { app } from "../../../../scripts/app.js";

function parseDimensions(str) {
  if (!str || typeof str !== "string") return [];
  const re = /^([1-9]\d*)\s*x\s*([1-9]\d*)$/i;
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
  const options = ["Auto", ...new Set(parseDimensions(text))];
  presetWidget.options.values = options;
  if (!options.includes(presetWidget.value)) presetWidget.value = "Auto";
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "basenc.ImageDimensions",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ImageDimensions") return;
    const dimensionsInput = nodeData.input?.required?.dimensions;
    const presetInput = nodeData.input?.required?.preset;
    if (presetInput && presetInput[0] === "COMBO") {
      presetInput[1] = presetInput[1] || {};
      const defaultDimensions =
        dimensionsInput?.[1]?.default ?? "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768";
      presetInput[1].values = ["Auto", ...new Set(parseDimensions(defaultDimensions))];
    }
  },

  nodeCreated(node) {
    if (node.comfyClass !== "ImageDimensions") return;
    const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
    if (dimensionsWidget) {
      const callback = dimensionsWidget.callback;
      dimensionsWidget.callback = function (value, ...args) {
        const result = callback?.call(this, value, ...args);
        updatePresetOptions(node);
        return result;
      };
    }
    updatePresetOptions(node);
  },
});
