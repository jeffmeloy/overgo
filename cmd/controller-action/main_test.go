package main

import (
	"bytes"
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/controlleraction"
	"overgo/internal/modelrecipe"
)

func TestPersistedDecisionMaterializationCommand(t *testing.T) {
	decision, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("decision"))
	if err != nil {
		t.Fatal(err)
	}
	closure, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("materialization"))
	if err != nil {
		t.Fatal(err)
	}
	first, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("first"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("second"))
	if err != nil {
		t.Fatal(err)
	}
	outputRoot := filepath.Join(t.TempDir(), "outputs")
	recordRoot := filepath.Join(t.TempDir(), "overgodb")
	original := materializeCandidateTrial
	t.Cleanup(func() { materializeCandidateTrial = original })
	calls := 0
	materializeCandidateTrial = func(
		_ context.Context,
		repository artifact.Repository,
		gotDecision artifact.ID,
		gotOutputRoot string,
	) (controlleraction.CandidateMaterializationResult, error) {
		if repository == nil || gotDecision != decision || gotOutputRoot != outputRoot {
			t.Fatalf("materialization invocation = (%v, %s, %q)", repository, gotDecision, gotOutputRoot)
		}
		calls++
		var commit artifact.CommitID
		if calls == 1 {
			commit[0] = 1
		}
		return controlleraction.CandidateMaterializationResult{
			Materialization: modelrecipe.CandidateMaterialization{ID: closure},
			Commit:          commit,
			Recovered:       calls > 1,
			Directories: map[artifact.ID]string{
				second: filepath.Join(outputRoot, "second"),
				first:  filepath.Join(outputRoot, "first"),
			},
		}, nil
	}
	args := []string{
		"-materialize-decision", decision.String(),
		"-record", recordRoot,
		"-output", outputRoot,
	}
	var output bytes.Buffer
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	text := output.String()
	ordered := []artifact.ID{first, second}
	slices.SortFunc(ordered, artifact.CompareID)
	if !strings.Contains(text, "candidate materialization: "+closure.String()) ||
		!strings.Contains(text, "arms=2") ||
		!strings.Contains(text, "sources were store-derived") ||
		strings.Index(text, ordered[0].String()) > strings.Index(text, ordered[1].String()) {
		t.Fatalf("command output = %q", text)
	}
	output.Reset()
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	if recovered := output.String(); !strings.Contains(recovered, " recovered arms=2") ||
		strings.Contains(recovered, "commit=") {
		t.Fatalf("recovery output = %q", recovered)
	}
}
