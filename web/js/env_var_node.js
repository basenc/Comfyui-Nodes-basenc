import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

let cachedEnvKeys = null;

async function fetchEnvKeys() {
  if (cachedEnvKeys) {
    return cachedEnvKeys;
  }

  try {
    const response = await api.fetchApi("/basenc/env_keys");
    const data = await response.json();
    if (data.keys && Array.isArray(data.keys)) {
      cachedEnvKeys = ["", ...data.keys];
      return cachedEnvKeys;
    }
  } catch (error) {
    console.error("[EnvVarNode] Failed to fetch environment keys:", error);
  }
  return [""];
}

app.registerExtension({
  name: "basenc.EnvVarNode",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "EnvVarNode") {
      const keys = await fetchEnvKeys();
      const envKeyInput = nodeData.input?.required?.env_key;

      if (envKeyInput && envKeyInput[0] === "COMBO") {
        envKeyInput[1] = envKeyInput[1] || {};
        envKeyInput[1].values = keys;
      }

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);

        const envKeyWidget = this.widgets?.find(w => w.name === "env_key");
        if (envKeyWidget && cachedEnvKeys) {
          envKeyWidget.options.values = cachedEnvKeys;
        }

        return result;
      };
    }
  }
});
