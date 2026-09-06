package webuilane

import "testing"

// TestReviewMeasures pins each criterion on a synthetic source: a swallow
// with a reviewed comment is not counted, a control inside a label or
// with a placeholder is named, a button with text is named, and the rest
// count once each.
func TestReviewMeasures(t *testing.T) {
	source := `const a = fetch("/x").catch(() => null);
const b = fetch("/y").catch(() => null) /* reviewed: the count route refuses for a relay */;
const c = load().catch(() => {});
if (!window.confirm("sure?")) return;
const title = prompt("title");
const one = el("input", { class: "text" });
const two = el("label", { class: "control" }, "Epochs", el("input", { type: "number" }));
const three = el("select", { class: "text" });
const four = el("input", { placeholder: "search" });
const five = el("input", { type: "file" });
const six = el("button", { class: "btn" });
const seven = el("button", { class: "btn", text: "go" });
const eight = el("button", { class: "btn" }, "go");
const nine = el("button", { class: "btn", text, onclick: () => go(item) });
const ten = el("button", { class: "btn", onclick: () => {
  go();
} }, "go");
const eleven = el("input", {
  class: "text", placeholder: "spans lines",
});
const twelve = el("button", { class: "btn" }, options.sendLabel || "send");
const styled = el("div", { style: "width:80px" });
const nested = a ? b ? 1 : 2 : 3;
setTimeout(tick, 1000); setInterval(poll, interval);
// TODO: later
`
	review := ReviewMeasures(source)
	want := Review{SilentFallbacks: 2, WindowDialogs: 2, UnnamedControls: 2, UnnamedButtons: 1, InlineStyles: 1, NestedTernaries: 1, TimerLiterals: 1, DebtMarkers: 1}
	if review != want {
		t.Fatalf("review = %+v, want %+v", review, want)
	}
	if (ReviewMeasures("const x = el(\"input\", { \"aria-label\": \"name\" });") != Review{}) {
		t.Fatal("a named input counted")
	}
}
