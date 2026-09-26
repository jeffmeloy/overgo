package webuilane

import (
	"strings"
	"testing"
)

// TestCensusClassifiesWithoutRegex holds the census to reading what the
// source says rather than what its text looks like: code-like text inside
// strings, comments, template literals and regular expressions counts for
// nothing, a call spanning lines is read whole, and a stylesheet's
// comments and strings split no rule, so a brace in either leaves the
// design tokens and the painted pairs intact.
func TestCensusClassifiesWithoutRegex(t *testing.T) {
	t.Parallel()
	quoted := "const help = \"write el(\\\"input\\\", {}) or setTimeout(go, 5) and TODO\";\n" +
		"const doc = `a ${value ? \"x\" : \"y\"} with style: \"red\" and .catch(() => null)`;\n" +
		"/* el(\"button\", {}) alert(\"x\") */\n" +
		"const pattern = /confirm\\(|style: \"/g;\n" +
		"// a ? b ? c : d : e\n"
	if review := ReviewMeasures(quoted); review != (Review{}) {
		t.Fatalf("text that only looks like code was counted: %+v", review)
	}
	spanning := "setTimeout(\n  tick,\n  1000\n);\nconst x = el(\n  \"button\",\n  { class: \"btn\" },\n);\n"
	if review := ReviewMeasures(spanning); review != (Review{TimerLiterals: 1, UnnamedButtons: 1}) {
		t.Fatalf("calls spanning lines were misread: %+v", review)
	}

	stylesheet := strings.Join([]string{
		"/* :root { --ink: #000000; } is not a declaration */",
		":root {",
		"  --ink: #eeeeee; --dim: #cccccc; --faint: #aaaaaa; --acc: #88ccff; --amber: #ffcc66; --ok: #88dd88; --err: #ff8888;",
		"  --bg0: #101010; --bg1: #181818; --bg2: #202020; --bg3: #282828; --s1: 4px; --t-h: 18px; --radius-s: 4px;",
		"}",
		".quote::before { content: \"}\"; color: var(--bg0); background: var(--acc); }",
		"@media (prefers-color-scheme: light) {",
		"  :root { --ink: #111111; --bg0: #ffffff; }",
		"}",
	}, "\n")
	document, err := ParseDesign(stylesheet)
	if err != nil {
		t.Fatal(err)
	}
	if document.Schemes[0].Colours["ink"] != "#eeeeee" || document.Schemes[1].Colours["ink"] != "#111111" ||
		document.Spacing["s1"] != "4px" || document.Type["t-h"] != "18px" || document.Radius["radius-s"] != "4px" {
		t.Fatalf("design tokens misread: %+v", document)
	}
	painted := false
	for _, pair := range document.Schemes[0].Contrast {
		painted = painted || pair.Foreground == "bg0" && pair.Background == "acc"
	}
	if !painted {
		t.Fatal("the pair a rule paints behind a quoted brace was lost")
	}
}
