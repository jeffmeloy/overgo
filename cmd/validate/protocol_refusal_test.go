package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/testutil"
)

// TestAProtocolTheModelCannotServeMeasuresNothing holds a benchmark cell whose evaluator refuses its protocol to a cell
// that measures nothing and fails nothing (ocr-domains): an evaluator that writes its refusal to the file the run names
// (evaluation.RefusalEnvironment) and ends is reported as a protocol the model's output cannot serve, with the
// refusal's words, while one that ends without writing one is a failure as before. Unlimited-OCR's chat-template
// mmlu-pro had failed every fill at 3a7b3877d, the OCR decoder answering " 1. The answer to the question".
func TestAProtocolTheModelCannotServeMeasuresNothing(t *testing.T) {
	t.Parallel()
	standIn := t.TempDir()
	testutil.WriteTextFile(t, standIn, "go.mod", "module standin\n\ngo 1.26\n", 0o600)
	testutil.WriteTextFile(t, standIn, "main.go", "package main\n\nimport \"os\"\n\n"+
		"func main() {\n\tif path := os.Getenv(\""+evaluation.RefusalEnvironment+"\"); path != \"\" && len(os.Args) > 1 {\n"+
		"\t\t_ = os.WriteFile(path, []byte(os.Args[1]), 0o600)\n\t}\n\tos.Exit(1)\n}\n", 0o600)
	refusal := "evaluation: chat-template protocol refused: none of the first 8 generated answers named a candidate"
	var output bytes.Buffer
	err := acquireEvaluation(t.Context(), &output, standIn, "go", []string{"run", ".", refusal})
	if !errors.Is(err, errProtocolCannotServe) || !strings.Contains(err.Error(), refusal) {
		t.Errorf("an evaluator that refused its protocol is reported as %v; want a protocol the model's output cannot serve, with its words", err)
	}
	if err := acquireEvaluation(t.Context(), &output, standIn, "go", []string{"run", "."}); err == nil || errors.Is(err, errProtocolCannotServe) {
		t.Errorf("an evaluator that ended without a refusal is reported as %v; want a failure", err)
	}
}
