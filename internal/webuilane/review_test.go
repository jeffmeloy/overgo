package webuilane

import (
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
)

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
	// four and eleven carry only a placeholder, which names nothing.
	want := Review{SilentFallbacks: 2, WindowDialogs: 2, UnnamedControls: 4, UnnamedButtons: 1, InlineStyles: 1, NestedTernaries: 1, TimerLiterals: 1, DebtMarkers: 1}
	if review != want {
		t.Fatalf("review = %+v, want %+v", review, want)
	}
	var combined Review
	combined.Add(review)
	combined.Add(review)
	if combined != ReviewMeasures(source+source) {
		t.Fatal("shared aggregation differs from measuring both sources")
	}
	if (ReviewMeasures("const x = el(\"input\", { \"aria-label\": \"name\" });") != Review{}) {
		t.Fatal("a named input counted")
	}
}

// webuiReviewCeiling: the review criteria ReviewMeasures counts, each at its
// measured value when the census landed (webui-quality-census); they only tighten.
var webuiReviewCeiling = Review{
	SilentFallbacks: 0, WindowDialogs: 0, UnnamedControls: 0, UnnamedButtons: 0,
	InlineStyles: 0, NestedTernaries: 0, TimerLiterals: 0, DebtMarkers: 0,
}

// clientJavaScript returns every JavaScript source of the served client by its path under the client's directory.
//   - the client's directory = the directory of StylesheetPath, which internal/server embeds whole
//   - a directory that holds no JavaScript is refused, because a ratchet over nothing holds nothing
func clientJavaScript(t *testing.T) map[string]string {
	t.Helper()
	client := os.DirFS(filepath.Join("..", "..", filepath.FromSlash(path.Dir(StylesheetPath))))
	sources := map[string]string{}
	if err := fs.WalkDir(client, ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(name, ".js") {
			data, err := fs.ReadFile(client, name)
			if err != nil {
				return err
			}
			sources[name] = string(data)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(sources) == 0 {
		t.Fatalf("%s holds no JavaScript source", path.Dir(StylesheetPath))
	}
	return sources
}

// TestWebUIReviewRatchet holds the client to the review criteria at their
// measured values: a source that adds a silent fallback, a dialog, an
// unnamed control, inline styling, a nested ternary, a timer literal or a
// debt marker fails here, and each ceiling only tightens as rows pay it.
func TestWebUIReviewRatchet(t *testing.T) {
	t.Parallel()
	var review Review
	largest := 0
	for name, source := range clientJavaScript(t) {
		measured := ReviewMeasures(source)
		if measured.SilentFallbacks+measured.WindowDialogs+measured.UnnamedControls+measured.UnnamedButtons > 0 {
			t.Logf("%s: %+v", name, measured)
		}
		review.Add(measured)
		largest = max(largest, strings.Count(source, "\n"))
	}
	for _, check := range []struct {
		name           string
		value, ceiling int
	}{
		{"silent fallbacks", review.SilentFallbacks, webuiReviewCeiling.SilentFallbacks},
		{"window dialogs", review.WindowDialogs, webuiReviewCeiling.WindowDialogs},
		{"unnamed controls", review.UnnamedControls, webuiReviewCeiling.UnnamedControls},
		{"unnamed buttons", review.UnnamedButtons, webuiReviewCeiling.UnnamedButtons},
		{"inline styles", review.InlineStyles, webuiReviewCeiling.InlineStyles},
		{"nested ternaries", review.NestedTernaries, webuiReviewCeiling.NestedTernaries},
		{"timer literals", review.TimerLiterals, webuiReviewCeiling.TimerLiterals},
		{"debt markers", review.DebtMarkers, webuiReviewCeiling.DebtMarkers},
	} {
		if check.value > check.ceiling {
			t.Errorf("%s = %d, ceiling %d", check.name, check.value, check.ceiling)
		}
	}
	t.Logf("review ratchet: %+v largest=%d", review, largest)
}
