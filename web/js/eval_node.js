import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

function promptNode(graph, id) {
  const parts = id.split(":");
  for (const part of parts.slice(0, -1)) graph = graph?.getNodeById(Number(part))?.subgraph;
  return graph?.getNodeById(Number(parts.at(-1)));
}

app.registerExtension({
  name: "basenc.Eval",
  nodeCreated(node) {
    if (node.comfyClass !== "Eval") return;
    node.addWidget("button", "Add output", null, () => {
      const count = node.outputs.length;
      if (count < 255) {
        node.outputs[0].name = node.outputs[0].label = "Result 1";
        node.addOutput(`Result ${count + 1}`, "*");
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
      }
    }, { serialize: false });
  },
  setup() {
    const graphToPrompt = app.graphToPrompt;
    app.graphToPrompt = async function (...args) {
      const prompt = await graphToPrompt.apply(this, args);
      const evals = Object.entries(prompt.output).filter(([, node]) => node.class_type === "Eval");
      if (evals.length) {
        const response = await api.fetchApi("/basenc/eval_outputs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(evals.map(([id]) => ({ count: promptNode(app.graph, id).outputs.length }))),
        });
        if (!response.ok) throw new Error(await response.text());
        const types = await response.json();
        evals.forEach(([, node], index) => { node.class_type = types[index]; });
      }
      return prompt;
    };
  },
});
