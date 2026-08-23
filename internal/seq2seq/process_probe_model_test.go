//go:build modeltest

package seq2seq

import (
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
	"overgo/internal/textgeneration"
)

func TestSingleTokenProcessProbe(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGenerator(filepath.Join(roots.Models, "needle"))
	if err != nil {
		t.Fatal(err)
	}
	output, err := generateThroughRecipe(t, generator, textgeneration.Request{Text: "hello", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("NEEDLE_PROCESS_PROBE output=%q\n", output)
}
