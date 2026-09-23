//go:build modeltest

package seq2seq

import (
	"fmt"
	"testing"

	"overgo/internal/modeltest"
	"overgo/internal/textgeneration"
)

func TestSingleTokenProcessProbe(t *testing.T) {
	generator, err := LoadGenerator(modeltest.Directory(t, "needle"))
	if err != nil {
		t.Fatal(err)
	}
	output, err := generateThroughRecipe(t, generator, textgeneration.Request{Text: "hello", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("NEEDLE_PROCESS_PROBE output=%q\n", output)
}
