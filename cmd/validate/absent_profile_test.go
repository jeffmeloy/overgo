package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"overgo/internal/hfrepo"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// TestACellWithNoProfileRefusesThePreflight holds a cell whose source no profile
// registers to a coverage gap: nothing names a lane that owns it, so the preflight
// reports it as refused, beside a cell that is ready, and does not pass over it.
func TestACellWithNoProfileRefusesThePreflight(t *testing.T) {
	t.Parallel()
	_, refusal := modelrecipe.ResolveGenerationSource(recipe.TaskImageGen, hfrepo.Identity{ModelType: "registered-nowhere"})
	if !errors.Is(refusal, modelrecipe.ErrGenerationSourceAbsent) {
		t.Fatalf("an unregistered source = %v", refusal)
	}
	crossed := fmt.Errorf("exit=1: recipe: %s\nexit status 1", refusal)
	cell := func(name string) Selected {
		return Selected{ModelValidation: ModelValidation{ModelName: name, Kind: recipe.TaskImageGen, Validation: "image-proof", Location: "C:/models/" + name + "/weights.safetensors", Directory: "C:/models/" + name}}
	}
	run := func(_ context.Context, _ io.Writer, _, _ string, args []string, _ ...string) error {
		if strings.Contains(strings.Join(args, " "), "registered-nowhere") {
			return crossed
		}
		return nil
	}
	var out bytes.Buffer
	err := preflightCells(t.Context(), &out, ".", "C:/stores/selected", []Selected{cell("registered-nowhere"), cell("ready")}, run)
	if err == nil || !strings.Contains(err.Error(), "1 of 2 cell(s) would be refused") || !strings.Contains(err.Error(), "registered-nowhere") {
		t.Fatalf("a cell no profile registers passed the preflight: %v\n%s", err, out.String())
	}
}
