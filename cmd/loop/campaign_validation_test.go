package main

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCampaignValidationReadsFreshFailure(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	writer, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	reader, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	world := &execWorld{stopStore: reader}
	if err := world.awaitValidation(); err != nil {
		t.Fatal(err)
	}
	pending, err := runrecord.NewGateLaneObligation(strings.Repeat("c", 40), testutil.ArtifactID(t, artifact.KindEvidence, "preparation"), testutil.ArtifactID(t, artifact.KindEvidence, "gate"), []string{"webui-lane"}, []string{"docs/plan.json"}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	publish := func(value runrecord.GateLaneObligation, previous *artifact.ID) {
		t.Helper()
		batch, err := value.Batch(previous)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	publish(pending, nil)
	running, err := pending.Transition(runrecord.LaneObligationRunning, artifact.ID{}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	publish(running, &pending.ID)
	failed, err := running.Transition(runrecord.LaneObligationFailed, testutil.ArtifactID(t, artifact.KindEvidence, "actual-failure"), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	publish(failed, &running.ID)
	// This temp directory has no cmd/gate. Success proves a recorded failure
	// was read through the old handle and was not blindly executed again.
	if err := world.awaitValidation(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(world.validationDebt, failed.Outcome.String()) {
		t.Fatal("next worker lost the failed result")
	}
	superseded, err := failed.Transition(runrecord.LaneObligationSuperseded, testutil.ArtifactID(t, artifact.KindEvidence, "repair"), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	publish(superseded, &failed.ID)
	if err := world.awaitValidation(); err != nil || world.validationDebt != "" {
		t.Fatalf("completed repair retained debt: %s %v", world.validationDebt, err)
	}
}

func TestCampaignWaitHonorsCancellation(t *testing.T) {
	root := t.TempDir()
	owner, err := authoritylock.Acquire(root)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if err := authoritylock.Wait(ctx, root); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled wait: %v", err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := authoritylock.Wait(t.Context(), root); err != nil {
		t.Fatal(err)
	}
}
