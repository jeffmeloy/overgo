package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/hfrepo"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// TestACellWhoseRuntimeIsNotHereWaitsOnItsLane holds a cell whose capability
// this checkout registers no generation source profile for, MOSS-Image's image
// cell on the trunk, to waiting on the lane whose runtime it is: the producer's
// refusal, read as its text across the process that ran it, is named apart from
// the failures and refuses nothing, while a cell refused for anything else
// still refuses the preflight.
func TestACellWhoseRuntimeIsNotHereWaitsOnItsLane(t *testing.T) {
	t.Parallel()
	_, refusal := modelrecipe.ResolveGenerationSource(recipe.TaskImageGen, hfrepo.Identity{ModelType: "registered-nowhere"})
	crossed := fmt.Errorf("exit=1: recipe: %s\nexit status 1", refusal)
	if !modelrecipe.OtherLane(refusal) || !modelrecipe.OtherLane(crossed) || modelrecipe.OtherLane(errors.New("exit=1: recipe: a weight file is missing")) || modelrecipe.OtherLane(nil) {
		t.Fatalf("the refusal of a source no profile registers reads as another lane's: %t, across a process %t", modelrecipe.OtherLane(refusal), modelrecipe.OtherLane(crossed))
	}

	cell := func(name, validation, location string) Selected {
		return Selected{ModelValidation: ModelValidation{ModelName: name, Kind: recipe.TaskImageGen, Validation: validation, Location: location, Directory: filepath.Dir(filepath.Dir(location))}}
	}
	elsewhere := cell("MOSS-Image", "image-proof", "C:/models/MOSS-Image/transformer/diffusion_pytorch_model.safetensors")
	ready := cell("Krea-2-Turbo", "image-proof", "C:/models/Krea-2-Turbo/transformer/diffusion_pytorch_model.safetensors")
	run := func(producer func(location string) error) func(context.Context, io.Writer, string, string, []string, ...string) error {
		return func(_ context.Context, _ io.Writer, _, _ string, args []string, _ ...string) error {
			return producer(strings.Join(args, " "))
		}
	}
	waiting := func(arguments string) error {
		if strings.Contains(arguments, "MOSS-Image") {
			return crossed
		}
		return nil
	}
	var out bytes.Buffer
	if err := preflightCells(t.Context(), &out, ".", "C:/stores/selected", []Selected{elsewhere, ready}, run(waiting)); err != nil {
		t.Fatalf("a cell whose runtime is another lane's refused the preflight: %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "1 cell(s) wait on their lane, their runtime not on this checkout:\n  MOSS-Image image-proof: ") ||
		!strings.Contains(out.String(), "1 cell(s) ready") {
		t.Fatalf("the preflight did not name the waiting cell apart from the ready one:\n%s", out.String())
	}

	failing := func(arguments string) error {
		if strings.Contains(arguments, "Krea-2-Turbo") {
			return errors.New("exit=1: recipe: a weight file is missing")
		}
		return waiting(arguments)
	}
	out.Reset()
	err := preflightCells(t.Context(), &out, ".", "C:/stores/selected", []Selected{elsewhere, ready}, run(failing))
	if err == nil || !strings.Contains(err.Error(), "1 of 2 cell(s) would be refused") || strings.Contains(err.Error(), "MOSS-Image") {
		t.Fatalf("a cell refused for another reason beside the waiting one ended the preflight %v, want it alone refused", err)
	}
}
