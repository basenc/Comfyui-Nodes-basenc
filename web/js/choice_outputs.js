import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const NODE_TYPES = new Set(["TypeSafeChoice", "LlamaCppChoice"]);
const FIXED_OUTPUTS = 2;
let serializeGraph;
let refreshTimer;

function optionNames(node) {
  const input = node.inputs?.find(({ name }) => name === "choices");
  const link = node.graph?.links?.[input?.link];
  const criteria = link && node.graph.getNodeById(link.origin_id);
  if (input?.link && !criteria) return node.outputs.slice(FIXED_OUTPUTS).map(({ name }) => name);
  if (criteria?.comfyClass === "ChoiceCriteria") {
    const names = [];
    const visited = new Set();
    let current = criteria;
    while (current?.comfyClass === "ChoiceCriteria" && !visited.has(current.id)) {
      visited.add(current.id);
      names.unshift(current.widgets?.find(({ name }) => name === "name")?.value ?? "Choice");
      const previous = current.inputs?.find(({ name }) => name === "previous");
      const prior = node.graph.links[previous?.link];
      current = prior && node.graph.getNodeById(prior.origin_id);
    }
    return names;
  }
  return node.outputs.slice(FIXED_OUTPUTS).map(({ name }) => name);
}

function syncOutputs(node, names = optionNames(node)) {
  const previous = node.outputs.map(({ name }) => name);
  if (node.outputs[1]?.type !== "STRING") {
    node.addOutput("Probabilities JSON", "STRING");
    node.outputs.splice(1, 0, node.outputs.pop());
    node.outputs.slice(FIXED_OUTPUTS).forEach((output, index) => {
      for (const id of output.links ?? []) {
        const link = node.graph?.links?.[id];
        if (link) link.origin_slot = index + FIXED_OUTPUTS;
      }
    });
  }
  node.outputs[0].name = node.outputs[0].label = "Choice";
  node.outputs[1].name = node.outputs[1].label = "Probabilities JSON";
  while (node.outputs.length > names.length + FIXED_OUTPUTS) node.removeOutput(node.outputs.length - 1);
  names.forEach((name, i) => {
    const output = node.outputs[i + FIXED_OUTPUTS];
    if (output) output.name = output.label = name;
    else node.addOutput(name, "BOOLEAN");
  });
  const changed = names.length + FIXED_OUTPUTS !== previous.length ||
    node.outputs.some(({ name }, index) => name !== previous[index]);
  if (changed) node.setDirtyCanvas(true, true);
  return changed;
}

function refreshChoices(graph) {
  for (const node of graph?._nodes ?? []) {
    if (NODE_TYPES.has(node.comfyClass)) syncOutputs(node);
    if (node.subgraph) refreshChoices(node.subgraph);
  }
}

function promptNames(output, choice) {
  const names = [];
  let link = choice.inputs.choices;
  const visited = new Set();
  while (Array.isArray(link) && !visited.has(link[0])) {
    visited.add(link[0]);
    const node = output[link[0]];
    if (node?.class_type !== "ChoiceCriteria") break;
    names.unshift(node.inputs.name);
    link = node.inputs.previous;
  }
  return names;
}

function promptNode(graph, id) {
  const parts = id.split(":");
  for (const part of parts.slice(0, -1)) graph = graph?.getNodeById(Number(part))?.subgraph;
  return graph?.getNodeById(Number(parts.at(-1)));
}

function syncFromPrompt(prompt) {
  let changed = false;
  for (const [id, choice] of Object.entries(prompt.output)) {
    if (!NODE_TYPES.has(choice.class_type)) continue;
    const node = promptNode(app.graph, id);
    if (node) changed = syncOutputs(node, promptNames(prompt.output, choice)) || changed;
  }
  return changed;
}

function scheduleRefresh(graph) {
  queueMicrotask(() => refreshChoices(graph));
  clearTimeout(refreshTimer);
  if (serializeGraph) {
    refreshTimer = setTimeout(async () => syncFromPrompt(await serializeGraph.call(app)), 150);
  }
}

app.registerExtension({
  name: "basenc.ChoiceOutputs",
  nodeCreated(node) {
    if (NODE_TYPES.has(node.comfyClass) || node.comfyClass === "ChoiceCriteria") {
      const connected = node.onConnectionsChange;
      node.onConnectionsChange = function (...args) {
        const result = connected?.apply(this, args);
        scheduleRefresh(this.graph);
        return result;
      };
    }
    if (node.comfyClass === "ChoiceCriteria") {
      const widget = node.widgets?.find(({ name }) => name === "name");
      if (widget) {
        const callback = widget.callback;
        widget.callback = function (...args) {
          const result = callback?.apply(this, args);
          scheduleRefresh(node.graph);
          return result;
        };
      }
    }
  },
  loadedGraphNode(node) {
    if (NODE_TYPES.has(node.comfyClass)) {
      node.outputs.slice(FIXED_OUTPUTS).forEach((output, index) => {
        for (const id of output.links ?? []) {
          const link = node.graph?.links?.[id];
          if (link) link.origin_slot = index + FIXED_OUTPUTS;
        }
      });
      scheduleRefresh(node.graph);
    }
  },
  setup() {
    const graphToPrompt = app.graphToPrompt;
    serializeGraph = graphToPrompt;
    app.graphToPrompt = async function (...args) {
      refreshChoices(app.graph);
      let prompt = await graphToPrompt.apply(this, args);
      let choices = Object.entries(prompt.output).filter(([, node]) => NODE_TYPES.has(node.class_type));
      if (syncFromPrompt(prompt)) {
        prompt = await graphToPrompt.apply(this, args);
        choices = Object.entries(prompt.output).filter(([, node]) => NODE_TYPES.has(node.class_type));
      }
      if (choices.length) {
        const response = await api.fetchApi("/basenc/choice_outputs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(choices.map(([, choice]) => ({
            node_id: choice.class_type,
            count: promptNames(prompt.output, choice).length,
          }))),
        });
        if (!response.ok) throw new Error(await response.text());
        const types = await response.json();
        choices.forEach(([, choice], i) => { choice.class_type = types[i]; });
      }
      return prompt;
    };
  },
});
