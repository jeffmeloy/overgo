//go:build windows && modeltest

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// namedAway are scenes that each tend to hold one thing, and that thing, which a negative prompt names.
var namedAway = []struct{ prompt, thing string }{
	{"a busy city street at night", "cars"},
	{"a cozy living room", "a sofa"},
	{"a sandy beach at sunset", "people"},
	{"a breakfast table in the morning", "a cup of coffee"},
	{"a quiet harbor at dawn", "boats"},
	{"a garden in spring", "flowers"},
	{"a mountain village in winter", "snow"},
	{"a birthday party in a backyard", "balloons"},
	{"a farm field in summer", "cows"},
	{"an office desk", "a computer"},
	{"a child's bedroom", "toys"},
	{"a park in autumn", "trees"},
}

// TestATypedNegativePromptRemovesWhatItNames reads NAG's job on Krea-2-Turbo directly and small (common-controls/
// negative, the owner's direction of 2026-11-16): each of twelve scenes is drawn at 512 x 512 in Krea's own eight steps
// with two seeds, once with a negative prompt naming one thing the scene tends to hold and once without, and the served
// judge, answering as itself, says of each picture whether that thing is in it. A pair counts for the negative prompt
// where the thing is absent with it and present without, against it the other way, and ties where both agree; the
// pairs' exact paired test shows the thing absent more often with it beyond chance. Krea runs unguided, so before NAG
// answered a typed negative prompt, both pictures of a pair were one picture and every pair tied.
func TestATypedNegativePromptRemovesWhatItNames(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	krea := modeltest.Directory(t, "Krea-2-Turbo")
	store := testcheck.Must(overgodb.OpenContext(ctx, roots.Store))(t)
	defer func() { testcheck.NoError(t, store.Close()) }()
	capability := mediacapability.Catalog[recipe.TaskImageGen]
	source := testcheck.Must(capability.Resolve(ctx, krea))(t)
	selected := testcheck.Must(modelrecipe.ResolveActiveExecution(ctx, store, source.Inventory.Manifest.ID, recipe.TaskImageGen, modelrecipe.SessionWarm))(t)
	held := capabilityruntime.HoldSessions(ctx)
	root := t.TempDir()
	var items []evaluation.TIFAItem
	pictures := map[string]string{}
	for scene, asked := range namedAway {
		for seed := 1; seed <= 2; seed++ {
			for _, negative := range []string{"", asked.thing} {
				request := map[string]any{"prompt": asked.prompt, "width": 512, "height": 512, "steps": 8, "seed": seed}
				id := fmt.Sprintf("scene%02d-seed%d-without", scene, seed)
				if negative != "" {
					request["negative_prompt"], id = negative, fmt.Sprintf("scene%02d-seed%d-with", scene, seed)
				}
				raw := testcheck.Must(json.Marshal(request))(t)
				output := testcheck.Must(capability.Execute(capabilityruntime.FreshRun(held), store, krea, selected, string(raw)))(t)
				content := testcheck.Must(mediacapability.OutputContent(capabilityruntime.Unwrap(output)))(t)
				testcheck.NoError(t, os.WriteFile(filepath.Join(root, id+".png"), content.Data, 0o644))
				pictures[id] = id + ".png"
				items = append(items, evaluation.TIFAItem{ID: id, Caption: asked.prompt, Questions: []evaluation.TIFAQuestion{
					{Question: fmt.Sprintf("Is there %s in the picture?", asked.thing), Choices: []string{"yes", "no"}, Answer: "no"},
				}})
			}
		}
	}
	testcheck.Must(capabilityruntime.ReleaseIdleSessions(ctx))(t)

	// The judge answers as itself, a vision-language model shown fixed media, so the suite names no judge.
	suite := testcheck.Must(evaluation.BuildTIFAQuestions(items, pictures, "the served judge"))(t)
	suite.Judge = ""
	judge := openCalibration(t, ctx, roots, root)
	compiled := testcheck.Must(judge.session.campaign.CompileSuite(ctx, testcheck.Must(json.Marshal(suite))(t)))(t)
	result := testcheck.Must(judge.session.campaign.Evaluate(ctx, compiled))(t)
	outcomes, read := evaluation.CaseOutcomes(ctx, judge.answers, result.Evaluation)
	if !read || len(outcomes) != len(suite.Cases) {
		t.Fatalf("the judge's %d answers to the %d cases cannot be read", len(outcomes), len(suite.Cases))
	}
	absent := map[string]bool{}
	for _, testCase := range suite.Cases {
		absent[testCase.Group] = outcomes[testCase.Name]
	}
	var wins, losses, ties uint64
	for scene := range namedAway {
		for seed := 1; seed <= 2; seed++ {
			with, without := absent[fmt.Sprintf("scene%02d-seed%d-with", scene, seed)], absent[fmt.Sprintf("scene%02d-seed%d-without", scene, seed)]
			switch {
			case with && !without:
				wins++
			case without && !with:
				losses++
			default:
				ties++
			}
		}
	}
	tail := tensorstats.ExactPairedBinary(wins, losses, ties).Tail
	t.Logf("the named thing absent with the negative prompt and present without in %d pairs, the other way in %d, the same in %d; chance %.3g", wins, losses, ties, tail)
	if tail > calibrationLevel {
		t.Errorf("a typed negative prompt does not remove what it names beyond chance: %d pairs for it, %d against, %d tied, chance %.3g against %g", wins, losses, ties, tail, calibrationLevel)
	}
}
