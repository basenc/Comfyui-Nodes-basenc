import { app } from "../../../../scripts/app.js";

const { $el } = window.comfyAPI.utils;
const { ComfyButton } = window.comfyAPI.button;
const ROW_HEIGHT = 26;
const GAP = 2;

function argsWidget(node, name, spec) {
  const root = $el("div.flex.flex-col", { style: { gap: `${GAP}px` } });
  let rows = [];

  let serialized;

  function button(icon, tooltip, action) {
    return new ComfyButton({
      icon, tooltip, action, iconSize: 16,
      classList: "comfyui-button bg-input text-foreground shrink-0",
    }).element;
  }

  function field(label, value, onEdit) {
    return $el("input.bg-input.text-foreground.rounded-md.px-2.text-xs.min-w-0.flex-1", {
      ariaLabel: label, placeholder: label, title: label, value,
      oninput: event => onEdit(event.currentTarget.value),
    });
  }

  function save() {
    serialized = JSON.stringify(rows);
    widget.callback?.(serialized);
    node.graph?.change();
  }

  function resize() {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
  }

  function addRow() {

    rows.push(["", ""]);
    render();
    save();
    resize();
    root.children[rows.length - 1].querySelector("input").focus();
  }

  function render() {
    root.replaceChildren();
    const line = () => $el("div.flex.gap-1", {
      parent: root, style: { height: `${ROW_HEIGHT}px` },
    });
    rows.forEach((row, index) => {
      const target = line();
      for (const [column, label] of ["Flag", "Value"].entries()) {
        target.append(field(label, row[column], text => {
          row[column] = text;
          save();
        }));
      }
      target.append(button("close", `Remove argument ${index + 1}`, () => {
        rows.splice(index, 1);
        render();
        save();
        resize();
      }));
    });
    line().append(button("plus", "Add llama-server argument", addRow));
  }

  function setValue(text) {
    if (text === serialized) return;
    rows = JSON.parse(text);
    serialized = text;
    render();
  }

  setValue(spec[1].default);
  const height = () => ROW_HEIGHT + rows.length * (ROW_HEIGHT + GAP);
  const widget = node.addDOMWidget(name, "BASENC_LLAMA_ARGS", root, {
    getValue: () => serialized, setValue, margin: 0,
    getMinHeight: height, getMaxHeight: height, getHeight: height,
  });
  return { widget };
}

app.registerExtension({
  name: "basenc.LlamaArgs",
  getCustomWidgets: () => ({ BASENC_LLAMA_ARGS: argsWidget }),

});
