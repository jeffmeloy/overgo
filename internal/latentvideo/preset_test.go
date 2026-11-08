package latentvideo

import (
	"testing"

	"overgo/internal/testcheck"
)

// countingBackend is the layout fixture's backend, counting the network passes a trajectory runs.
type countingBackend struct {
	layoutFixtureBackend
	passes int
}

func (b *countingBackend) ForwardHead(patch, block, head []float32, branch any) ([]float32, error) {
	b.passes++
	return b.layoutFixtureBackend.ForwardHead(patch, block, head, branch)
}

// TestARequestRunsTheNamedPresetOfItsProfile holds a clip request to the recipe it names: on 2026-11-08 Wan's profile
// held one recipe, the authors' reference script's shift of 5, and its model card's shift of 8 was a code change. A
// request naming a preset runs its values over the authors' recipe, its own stated value standing, and comes out as
// the request that states them; one naming none runs the authors' recipe; one naming an undeclared preset is refused.
// At guidance 1 a trajectory runs one network pass a step, no unconditioned one.
func TestARequestRunsTheNamedPresetOfItsProfile(t *testing.T) {
	t.Parallel()
	profile := testcheck.Must(declaredProfile())(t)
	named := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox", Preset: "model-card"}))(t)
	reference := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox"}))(t)
	if named.Shift != 8 || reference.Shift != profile.Generation.Shift || named.Steps != reference.Steps || named.GuideScale != reference.GuideScale || named.Preset != "" {
		t.Fatalf("the model card's preset gave %+v against the authors' recipe %+v", named, reference)
	}
	if stated := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox", Preset: "model-card", Shift: 3}))(t); stated.Shift != 3 {
		t.Fatalf("a request's own shift gave way to its preset's: %+v", stated)
	}
	if _, err := profile.complete(WanRequest{Prompt: "a fox", Preset: "nonesuch"}); err == nil {
		t.Fatal("a request ran a preset its profile does not declare")
	}
	program, request := layoutFixture(1, 2, 4)
	for _, guidance := range []struct {
		scale  float64
		passes int
	}{{1, request.Steps}, {1.5, 2 * request.Steps}} {
		backend := &countingBackend{}
		request.GuideScale = guidance.scale
		testcheck.Must(program.DenoiseWithBackend(backend, request))(t)
		if backend.passes != guidance.passes {
			t.Fatalf("at guidance %g a %d-step trajectory ran %d network passes, want %d", guidance.scale, request.Steps, backend.passes, guidance.passes)
		}
	}
}
