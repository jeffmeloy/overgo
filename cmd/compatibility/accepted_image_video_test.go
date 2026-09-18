package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

type imageVideoCoverage struct {
	acceptedCoverage
	Cases []acceptedMediaCase
}

type acceptedMediaCase struct {
	Case                string
	Run, Output, Review artifact.ID
}

func checkAcceptedMediaCases(protocol imageVideoProtocol, coverage imageVideoCoverage) error {
	wanted := map[string]imageVideoCase{}
	for _, value := range protocol.Cases {
		if _, found := wanted[value.ID]; found {
			return errors.New("media acceptance: duplicate protocol case")
		}
		wanted[value.ID] = value
	}
	proofs := map[string]bool{}
	for _, proof := range coverage.Proofs {
		key := proof.Reference + "#" + proof.Check
		if proofs[key] {
			return errors.New("media acceptance: duplicate proof")
		}
		proofs[key] = true
	}
	for _, binding := range coverage.Cases {
		value, found := wanted[binding.Case]
		if !found || binding.Run.Kind() != artifact.KindRun || binding.Output.Kind() != artifact.KindOutput || binding.Review.Kind() != artifact.KindEvidence {
			return errors.New("media acceptance: absent, duplicate or invalid case")
		}
		bound := false
		for _, model := range coverage.Models {
			for _, cell := range model.Cells {
				if model.Model != value.Model || cell.Task != value.Task {
					continue
				}
				if bound || cell.Recipe != value.Recipe || len(cell.Proofs) == 0 {
					return errors.New("media acceptance: duplicate or changed activation")
				}
				for _, reference := range cell.Proofs {
					if !proofs[reference] {
						return fmt.Errorf("media acceptance: missing proof %s", reference)
					}
				}
				bound = true
			}
		}
		if !bound {
			return errors.New("media acceptance: case has no accepted activation")
		}
		delete(wanted, binding.Case)
	}
	if len(wanted) != 0 || len(coverage.Cases) == 0 {
		return errors.New("media acceptance: incomplete denominator")
	}
	return nil
}

func TestAcceptedImageVideoEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	var coverage imageVideoCoverage
	requireAcceptedDocument(t, root, "docs/verification/image-video-coverage.json", acceptedImageVideoSHA256, &coverage)
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		t.Fatal(err)
	}
	if err := checkAcceptedMediaCases(protocol, coverage); err != nil {
		t.Fatal(err)
	}
	store := requireAcceptedCoverage(t, root, coverage.acceptedCoverage, recipe.TaskImageGen, recipe.TaskVideoGen)
	for i, binding := range coverage.Cases {
		t.Run(binding.Case, func(t *testing.T) {
			value := protocol.Cases[i]
			if value.ID != binding.Case || len(value.Observations) != 1 {
				t.Fatal("case order or observation denominator changed")
			}
			// Corrected outputs supersede earlier encoding errors, not inputs or judges.
			value.Run, value.Outputs = binding.Run, []artifact.ID{binding.Output}
			run, err := runrecord.RequireRun(t.Context(), store, binding.Run)
			if err != nil {
				t.Fatal(err)
			}
			if err := checkImageVideoCaseRun(value, run); err != nil {
				t.Fatal(err)
			}
			content, found, err := artifact.ReadContent(t.Context(), store, binding.Output)
			if err != nil || !found {
				t.Fatalf("required output absent: %v", err)
			}
			if err := content.Validate(); err != nil {
				t.Fatal(err)
			}
			observed := value.Observations[0]
			observed.Artifact, observed.Delay = binding.Output, 0
			if observed.FPS != 0 {
				for frame := range observed.Frames {
					delay, err := media.GIFFrameDelay(observed.FPS, frame)
					if err != nil {
						t.Fatal(err)
					}
					observed.Delay += delay
				}
			}
			if err := checkImageVideoObservation(observed, content); err != nil {
				t.Fatal(err)
			}
			name, want, ok := sampleFile(content)
			actual, err := os.ReadFile(filepath.Join(root, "docs/media_samples", name))
			if !ok || err != nil || !bytes.Equal(actual, want) {
				t.Fatalf("exported sample missing or changed: %v", err)
			}
			if _, found, err := artifact.ReadContent(t.Context(), store, binding.Review); err != nil || !found {
				t.Fatalf("original visual review absent: %v", err)
			}
			for _, mutation := range []string{"input", "recipe", "output", "failed"} {
				changed := run
				switch mutation {
				case "input":
					changed.Inputs = nil
				case "recipe":
					changed.Recipe = artifact.ID{}
				case "output":
					changed.Outputs = nil
				case "failed":
					changed.Outcome = runrecord.OutcomeFailed
				}
				if checkImageVideoCaseRun(value, changed) == nil {
					t.Fatalf("changed %s received credit", mutation)
				}
			}
		})
	}
	for _, mutation := range []string{"omitted", "duplicate", "missing-proof", "changed-model"} {
		t.Run(mutation, func(t *testing.T) {
			changed := coverage
			switch mutation {
			case "omitted":
				changed.Cases = changed.Cases[1:]
			case "duplicate":
				changed.Cases = append(slices.Clone(changed.Cases), changed.Cases[0])
			case "missing-proof":
				changed.Proofs = nil
			case "changed-model":
				changed.Models = slices.Clone(changed.Models)
				changed.Models[0].Model = artifact.ID{}
			}
			if checkAcceptedMediaCases(protocol, changed) == nil {
				t.Fatal("invalid denominator received credit")
			}
		})
	}
	t.Logf("%d frozen cases, %d models, %d accepted phases; acquisition=0; original quality/resource scopes retained", len(coverage.Cases), len(coverage.Models), len(coverage.Proofs))
}

const acceptedImageVideoSHA256 = "88af07241fd07965fe65bb31b020c7f29fb400c7deadce60712ea54ae0d91d51"
