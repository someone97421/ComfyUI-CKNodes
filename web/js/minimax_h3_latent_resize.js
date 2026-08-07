import { app } from "../../../scripts/app.js";

const NODE_ID = "CKMiniMaxH3LatentResize";
const TARGET_WIDGETS = ["target_width", "target_height"];
const SCALE_WIDGETS = ["scale_by"];

function saveOriginalState(widget) {
    if (widget._ckH3ResizeOriginalState) return;
    widget._ckH3ResizeOriginalState = {
        computeSize: widget.computeSize,
        draw: widget.hasOwnProperty("draw") ? widget.draw : undefined,
        hidden: widget.hidden,
        optionsHidden: widget.options?.hidden,
        elementDisplay: widget.element?.style?.display,
    };
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    saveOriginalState(widget);
    const original = widget._ckH3ResizeOriginalState;

    widget.options ??= {};
    widget.hidden = !visible;
    widget.options.hidden = !visible;

    if (visible) {
        widget.computeSize = original.computeSize;
        if (original.draw === undefined) delete widget.draw;
        else widget.draw = original.draw;
        if (widget.element) {
            widget.element.style.display = original.elementDisplay ?? "";
        }
        delete widget.computedHeight;
    } else {
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        if (widget.element) widget.element.style.display = "none";
        widget.computedHeight = 0;
    }
}

function updateResizeModeWidgets(node) {
    const modeWidget = node.widgets?.find((widget) => widget.name === "resize_mode");
    if (!modeWidget) return;

    const targetMode = modeWidget.value === "target_resolution";
    for (const name of TARGET_WIDGETS) {
        setWidgetVisible(node.widgets?.find((widget) => widget.name === name), targetMode);
    }
    for (const name of SCALE_WIDGETS) {
        setWidgetVisible(node.widgets?.find((widget) => widget.name === name), !targetMode);
    }

    const computed = node.computeSize?.();
    if (computed && node.setSize) {
        node.setSize([node.size?.[0] ?? computed[0], computed[1]]);
    }
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "CKNodes.MiniMaxH3LatentResize.DynamicInputs",

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;

        const modeWidget = node.widgets?.find((widget) => widget.name === "resize_mode");
        if (!modeWidget) return;

        const originalCallback = modeWidget.callback;
        modeWidget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            updateResizeModeWidgets(node);
            return result;
        };

        // 创建新节点和加载已有工作流时，widgets_values 的恢复时机可能稍晚。
        requestAnimationFrame(() => updateResizeModeWidgets(node));
        setTimeout(() => updateResizeModeWidgets(node), 0);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = originalOnConfigure?.apply(this, args);
            requestAnimationFrame(() => updateResizeModeWidgets(this));
            return result;
        };
    },
});
