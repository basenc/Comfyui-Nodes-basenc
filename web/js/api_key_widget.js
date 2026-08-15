import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const NODE_TYPES = new Set(["CustomOpenAIResponse", "CustomOpenAICompletion"]);
const PENDING = "Storing API key...";
const CONFIGURED = Symbol("basencApiKeyConfigured");

async function storeSecret(secret) {
  const response = await api.fetchApi("/basenc/api_keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ secret }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()).reference;
}

function configureApiKey(node) {
  const widget = node.widgets?.find(({ name }) => name === "api_key");
  if (!widget || widget[CONFIGURED]) {
    return;
  }

  widget[CONFIGURED] = true;
  let internalUpdate = false;

  const secure = (value) => {
    if (
      internalUpdate ||
      !value ||
      value === PENDING ||
      (URL.canParse(value) && new URL(value).protocol === "basenc-secret:")
    ) {
      return;
    }

    internalUpdate = true;
    widget.value = PENDING;
    internalUpdate = false;
    node.setDirtyCanvas(true, true);

    storeSecret(value).then(
      (reference) => {
        internalUpdate = true;
        widget.value = reference;
        internalUpdate = false;
        node.setDirtyCanvas(true, true);
      },
      (error) => {
        internalUpdate = true;
        widget.value = "";
        internalUpdate = false;
        node.setDirtyCanvas(true, true);
        console.error("[basenc] Failed to store API key:", error);
      },
    );
  };

  widget.callback = secure;
  secure(widget.value);
}

app.registerExtension({
  name: "basenc.ApiKeyWidget",
  nodeCreated(node) {
    if (NODE_TYPES.has(node.comfyClass)) {
      configureApiKey(node);
    }
  },
  loadedGraphNode(node) {
    if (NODE_TYPES.has(node.comfyClass)) {
      configureApiKey(node);
    }
  },
});
