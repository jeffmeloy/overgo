package webuilane

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// readableContrast is the level-AA ratio body text must reach on its
// background, the threshold the layout audit measures pages against.
const readableContrast = 4.5

func derivedDesign(t *testing.T) (DesignDocument, []byte) {
	t.Helper()
	root := filepath.Join("..", "..")
	stylesheet, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(StylesheetPath)))
	if err != nil {
		t.Fatal(err)
	}
	document, err := ParseDesign(string(stylesheet))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeDesign(document)
	if err != nil {
		t.Fatal(err)
	}
	return document, encoded
}

// TestDesignDocumentMatchesStylesheet holds the published design document to
// the stylesheet it is derived from: style.css is the only token source, so
// a token changed there without `go run ./cmd/webui-lane -design` leaves the
// document stale and this test names the drift.
func TestDesignDocumentMatchesStylesheet(t *testing.T) {
	t.Parallel()
	document, derived := derivedDesign(t)
	published, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(DesignDocumentPath)))
	if err != nil {
		t.Fatalf("%s: %v (derive it with go run ./cmd/webui-lane -design)", DesignDocumentPath, err)
	}
	if !bytes.Equal(published, derived) {
		t.Fatalf("%s differs from %s; derive it again with go run ./cmd/webui-lane -design", DesignDocumentPath, StylesheetPath)
	}
	if len(document.Schemes) != 2 || len(document.Spacing) == 0 || len(document.Type) == 0 || len(document.Radius) == 0 {
		t.Fatalf("design document = %+v", document)
	}
	var decoded DesignDocument
	if err := json.Unmarshal(published, &decoded); err != nil || decoded.Source != StylesheetPath {
		t.Fatalf("published document decodes as %+v, %v", decoded.Source, err)
	}
}

// TestDesignMeasuresThePairsTheStylesheetPaints derives the pairs to measure
// from the stylesheet's own rules: the check once measured on-accent on the
// accent while buttons painted bg0 on it, and never measured ink on the
// translucent highlight of the active tab, so pairs read on screen went
// unmeasured. A translucent background counts at the worst surface under it.
func TestDesignMeasuresThePairsTheStylesheetPaints(t *testing.T) {
	t.Parallel()
	document, _ := derivedDesign(t)
	for _, scheme := range document.Schemes {
		measured := map[string]bool{}
		for _, pair := range scheme.Contrast {
			measured[pair.Foreground+" on "+pair.Background] = true
		}
		for _, painted := range []string{"on-accent on acc", "ink on acc-soft"} {
			if !measured[painted] {
				t.Errorf("%s: the stylesheet paints %s and the design does not measure it", scheme.Name, painted)
			}
		}
	}
	rule := ".x { color:var(--ink); /* note; */ background:var(--acc-soft); } .y { color:red; background:var(--bg0); }"
	if pairs := paintedPairs(rule); len(pairs) != 1 || pairs[0] != [2]string{"ink", "acc-soft"} {
		t.Fatalf("painted pairs of %q = %v", rule, pairs)
	}
	soft, _ := parseColour("rgba(255,255,255,.5)")
	if seen := soft.over(colour{alpha: 1}); seen.channels[0] != 127.5 || seen.alpha != 1 {
		t.Fatalf("half white over black = %+v", seen)
	}
}

// TestCraftFloorTokenContrast measures every text token on every surface,
// and the accent's own text on the accent, in both colour schemes: each pair
// reads at the level-AA ratio. Dark error red on the raised surfaces and
// light success green on the highest one once fell short.
func TestCraftFloorTokenContrast(t *testing.T) {
	t.Parallel()
	document, _ := derivedDesign(t)
	pairs := 0
	for _, scheme := range document.Schemes {
		for _, pair := range scheme.Contrast {
			pairs++
			if pair.Ratio < readableContrast {
				t.Errorf("%s: %s on %s reads at %.2f:1, below %.1f:1", scheme.Name, pair.Foreground, pair.Background, pair.Ratio, readableContrast)
			}
		}
	}
	if least := 2 * len(textTokens) * len(surfaceTokens); pairs <= least {
		t.Fatalf("measured %d pairs, want every text token on every surface (%d) and the painted pairs beside", pairs, least)
	}
	white, okWhite := parseColour("#fff")
	black, okBlack := parseColour("#000000")
	if !okWhite || !okBlack {
		t.Fatal("short and long hex colours must parse")
	}
	if ratio := contrast(white, black); ratio != 21 {
		t.Fatalf("white on black = %v, want 21", ratio)
	}
}
