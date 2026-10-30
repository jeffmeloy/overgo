package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/recipe"
)

// TestPreflightRefusesAnUnmeetableCell holds the preflight to running each cell's command as its
// measurement would, told to stop where the measurement begins, and to naming every cell that
// would be refused before any is measured. The runner stands in for the producers: the guard's
// refuses, as Flash-Next's did for want of a bound short-prompt benchmark, and every other
// passes. A cell with no bound command is refused as the run refuses it, a training proof's
// checkpoint goes to the preflight's scratch and not where the cell names, a cell measured by a
// go test is named as checked only when measured, and no lane runs.
func TestPreflightRefusesAnUnmeetableCell(t *testing.T) {
	t.Parallel()
	cell := func(name, validation, location string) Selected {
		return Selected{ModelValidation: ModelValidation{ModelName: name, Kind: recipe.TaskInference, Validation: validation, Location: location}}
	}
	trained := cell("trained", trainingProof, "C:/models/trained/model.safetensors")
	trained.Objective = "objective.registered.text-text"
	oracle := cell("oracle", projectionProof, "")
	oracle.Grounding = protocolOracle
	cells := []Selected{
		cell("ready", servingValidation, "C:/models/ready.gguf"),
		cell("unmet", guardValidation, "C:/models/unmet.gguf"),
		cell("unbound", servingValidation, ""),
		trained, oracle,
	}
	var ran []string
	run := func(_ context.Context, _ io.Writer, _, name string, args []string, environment ...string) error {
		ran = append(ran, strings.Join(args, " "))
		if !slices.Contains(environment, clioptions.PreflightEnvironment+"=1") {
			t.Errorf("%s %q ran without the preflight in its environment %q", name, args, environment)
		}
		if at := slices.Index(args, "-out"); at >= 0 && args[at+1] == "checkpoint" {
			t.Errorf("the training proof's checkpoint goes where the cell names, %q, want the preflight's scratch", args[at+1])
		}
		if slices.Contains(args, "./cmd/longform") {
			return errors.New("no bound finite positive short-prompt benchmark")
		}
		return nil
	}
	var out bytes.Buffer
	err := preflightCells(t.Context(), &out, ".", "C:/stores/selected", cells, run)
	if err == nil || !strings.Contains(err.Error(), "2 of 5 cell(s) would be refused") ||
		!strings.Contains(err.Error(), "unmet guard: no bound finite positive short-prompt benchmark") || !strings.Contains(err.Error(), "unbound "+servingValidation) {
		t.Fatalf("the preflight ended %v, want the guard's refusal and the unbound cell named\n%s", err, out.String())
	}
	if len(ran) != 3 || !strings.Contains(ran[0], "./cmd/recipe measure") || !strings.Contains(ran[1], "./cmd/longform") || !strings.Contains(ran[2], "./cmd/train") {
		t.Errorf("the preflight ran %q, want the three bound producers that are no go test, in the cells' order", ran)
	}
	if !strings.Contains(out.String(), "checked only when measured: oracle "+projectionProof) {
		t.Errorf("the preflight did not name the cell a go test measures\n%s", out.String())
	}
}
