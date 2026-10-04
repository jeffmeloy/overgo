//go:build modeltest

package seq2seq

import (
	"fmt"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/textgeneration"
)

// TestSingleTokenProcessProbe is the measured child of the Needle process
// comparison: it loads the model its parent resolved through the store and
// generates one token, so the measured peak is the model's and not a replay
// of the store that registered it.
func TestSingleTokenProcessProbe(t *testing.T) {
	directory := os.Getenv("OVERGO_NEEDLE_MODEL")
	if directory == "" {
		testskip.NotApplicable(t, "child process probe requires OVERGO_NEEDLE_MODEL")
	}
	generator, err := LoadGenerator(directory)
	if err != nil {
		t.Fatal(err)
	}
	output, err := generateThroughRecipe(t, generator, textgeneration.Request{Text: "hello", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("NEEDLE_PROCESS_PROBE output=%q\n", output)
}
