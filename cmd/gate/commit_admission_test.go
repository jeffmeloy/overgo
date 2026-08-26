package main

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

func TestCommitAdmissionUsesManifestPlan(t *testing.T) {
	runner := func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil }
	checks := []automationcheck.Check{
		{Descriptor: automationcheck.Descriptor{Name: "verify", Phase: runrecord.PhaseTest, Always: true}, Run: runner},
		{Descriptor: automationcheck.Descriptor{
			Name: "commit", Phase: runrecord.PhasePackage, Always: true, Dependencies: []string{"verify"},
		}, Run: runner},
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	manifest, err := automationcheck.BindManifestPlan(
		base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64),
		automationcheck.Surface{Identity: "surface"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("inputs"))
	bound, err := automationcheck.BindManifestExecution(manifest, invocations[0], []artifact.ID{input})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := automationcheck.Run(context.Background(), bound)
	if err != nil {
		t.Fatal(err)
	}
	terminal := map[string]automationcheck.Evidence{"verify": evidence}
	if err := validateManifestCommitAdmission(manifest, terminal); err != nil {
		t.Fatal(err)
	}
	foreign := evidence
	foreign.Authority = &automationcheck.ExecutionAuthority{
		Plan: manifest.ID, Definition: invocations[1].ID, Inputs: []artifact.ID{input},
	}
	if err := validateManifestCommitAdmission(manifest, map[string]automationcheck.Evidence{"verify": foreign}); err == nil {
		t.Fatal("foreign definition evidence admitted")
	}
	if err := validateManifestCommitAdmission(manifest, nil); err == nil {
		t.Fatal("missing terminal evidence admitted")
	}
}

func TestOutcomeDistinctions(t *testing.T) {
	passed := automationcheck.Evidence{DurationNS: 1}
	inapplicable := passed
	inapplicable.Inapplicable = true
	reused := passed
	reused.Reused = true
	failed := gateEvidenceRecord("failed", runrecord.PhaseTest, passed, errors.New("failed"), "")
	if got := gateEvidenceRecord("inapplicable", runrecord.PhaseTest, inapplicable, nil, "").Outcome; got != runrecord.StepInapplicable {
		t.Fatalf("inapplicable outcome = %s", got)
	}
	if got := gateEvidenceRecord("reused", runrecord.PhaseTest, reused, nil, "").Outcome; got != runrecord.StepReused {
		t.Fatalf("reused outcome = %s", got)
	}
	if failed.Outcome != runrecord.StepFailed || runrecord.StepSkipped == runrecord.StepInapplicable {
		t.Fatalf("failed/skipped/inapplicable collapsed: %+v", failed)
	}
}
