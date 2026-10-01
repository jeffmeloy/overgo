//go:build modeltest

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestAcceptedModalityCoverage(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT for retained modality coverage")
	}
	t.Run("case-denominators", TestProtocolCaseProjection)
	t.Run("resource-denominators", TestResourceCaseProjection)
	// The report is rendered on demand, never committed: its projection is
	// held to the declared protocols here, not to a stored copy.
	output := acceptedProtocolCoverageProjection(t)
	root := testutil.RepoRoot(t)
	var projection mediaProjection
	if err := json.Unmarshal(output.JSON, &projection); err != nil {
		t.Fatal(err)
	}
	var generation imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &generation); err != nil {
		t.Fatal(err)
	}
	for _, declared := range generation.Cases {
		matches := 0
		for _, model := range projection.Coverage.Models {
			for _, p := range model.Protocols {
				if p.Name == declared.ID && model.Model == declared.Model && p.Recipe == declared.Recipe && p.Unit == "generation case" && p.Required != nil && *p.Required == 1 && p.Recorded != nil && *p.Recorded == 1 {
					matches++
				}
			}
		}
		if matches != 1 {
			t.Fatalf("generation case %s: projected %d times", declared.ID, matches)
		}
	}
	var source modelValidationSpecification
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/e4b-validation.json"), &source); err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(projection.Coverage.Models, func(row modalityModelProjection) bool { return row.Model == source.Model })
	if index < 0 {
		t.Fatal("multimodal inference model omitted")
	}
	protocols := projection.Coverage.Models[index].Protocols
	required := e4bValidationCells()
	checkNames := func(values []modalityProtocolProjection) bool {
		if len(values) != len(required) {
			return false
		}
		seen := map[string]bool{}
		for _, value := range values {
			if seen[value.Name] || !slices.Contains(required, value.Name) {
				return false
			}
			seen[value.Name] = true
		}
		return true
	}
	if !checkNames(protocols) || checkNames(protocols[1:]) {
		t.Fatal("modality protocol denominator differs")
	}
	for _, value := range protocols {
		if value.Required == nil || value.Recorded == nil || *value.Required != *value.Recorded {
			t.Fatalf("case denominator unbound: %s", value.Name)
		}
		if strings.HasPrefix(value.Name, "resources/") && value.RepairOwner != "device-memory-retention/do" {
			t.Fatal("current resource gap lost its owner")
		}
	}
	for name, mutate := range map[string]func(*modalityProtocolProjection){
		"missing-mode":        func(p *modalityProtocolProjection) { p.InputMode = "" },
		"missing-source":      func(p *modalityProtocolProjection) { p.Source = "foreign" },
		"missing-denominator": func(p *modalityProtocolProjection) { p.Required = nil },
		"missing-repair":      func(p *modalityProtocolProjection) { p.RepairOwner = "" },
		"contradictory-count": func(p *modalityProtocolProjection) { n := *p.Recorded + 1; p.Recorded = &n },
	} {
		t.Run(name, func(t *testing.T) {
			changed := protocols[0]
			mutate(&changed)
			if checkModalityProtocols([]modalityProtocolProjection{changed}) == nil {
				t.Fatal("invalid protocol received case credit")
			}
		})
	}
	t.Log("Current catalog and original protocol scopes reconciled; required examples, recorded examples, generation cases and recovery actions remain separate. No model acquisitions.")
}

func TestAcceptedTextVisionEvidence(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	root := testutil.RepoRoot(t)
	var coverage acceptedCoverage
	requireAcceptedDocument(t, root, "docs/verification/text-vision-coverage.json", "ec07d869988aa9cdbe9ea0e3d8a41d6408188aa80acce418d9b3f7f6ae27c381", &coverage)
	requireAcceptedCoverage(t, root, coverage, recipe.TaskInference, recipe.TaskProjection, recipe.TaskVQA, recipe.TaskEmbedding, recipe.TaskRerank)
}

func TestAcceptedSpecializedTaskEvidence(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	root := testutil.RepoRoot(t)
	var coverage acceptedCoverage
	requireAcceptedDocument(t, root, "docs/verification/specialized-coverage.json", "e918302bf88dfa0b831391835a4b515dd40b96d425f52624e82d83b342198d63", &coverage)
	requireAcceptedCoverage(t, root, coverage, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq, recipe.TaskSpeech, recipe.TaskTranscription, recipe.TaskGeneration, recipe.TaskAlignment, recipe.TaskDiarization, recipe.TaskActivityDetection, recipe.TaskAudioConversion, recipe.TaskAudioGeneration)
}
