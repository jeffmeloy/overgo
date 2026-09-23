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
	if want := 2 * (len(textTokens)*len(surfaceTokens) + 1); pairs != want {
		t.Fatalf("measured %d pairs, want %d", pairs, want)
	}
	if _, err := contrastRatio("#fff", "#000"); err != nil {
		t.Fatal(err)
	}
	if ratio, _ := contrastRatio("#ffffff", "#000000"); ratio != 21 {
		t.Fatalf("white on black = %v, want 21", ratio)
	}
}
