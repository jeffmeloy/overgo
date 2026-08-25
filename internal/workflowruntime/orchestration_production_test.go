package workflowruntime

import (
	"context"
	"errors"
	"testing"
	"time"

	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

func TestOrchestrationRestartRecovery(t *testing.T) {
	anchor := time.Date(2026, time.August, 24, 8, 0, 0, 0, time.UTC)
	fixture := newScheduledAutomationRuntimeFixture(t, anchor)
	defer fixture.store.Close()
	firstManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	first := fixture.runtime(t, firstManager, generationAutomationAdapters(t, true, nil))
	execution, fired, err := first.ScheduleAutomation(
		context.Background(), fixture.name, fixedAutomationClock{now: anchor.Add(time.Hour)}, fixture.inputs,
	)
	if err != nil || !fired || !execution.Claim.Valid() {
		t.Fatalf("restart schedule admission=(%+v, %t, %v)", execution, fired, err)
	}
	failed, err := firstManager.Wait(context.Background(), execution.Operation)
	firstManager.Close()
	if err != nil || failed.State != operation.StateFailed || failed.Run == nil {
		t.Fatalf("restart prerequisite=(%+v, %v)", failed, err)
	}

	secondManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer secondManager.Close()
	adapters := generationAutomationAdapters(t, false, nil)
	adapters[workflowrecipe.ModuleTokenize] = AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
		return nil, errors.New("completed stage reran after restart")
	})
	second := fixture.runtime(t, secondManager, adapters)
	recovered, err := second.RecoverScheduledAutomation(context.Background(), execution.Claim, fixture.inputs)
	if err != nil || recovered.Operation != execution.Operation || recovered.Plan.ID != execution.Plan.ID || recovered.Claim != execution.Claim {
		t.Fatalf("restart recovery identity=(%+v, %v)", recovered, err)
	}
	completed, err := secondManager.Wait(context.Background(), recovered.Operation)
	if err != nil || completed.State != operation.StateCompleted || completed.Run == nil || len(completed.Outputs) == 0 {
		t.Fatalf("restart recovery result=(%+v, %v)", completed, err)
	}
	for _, node := range fixture.definition.Nodes {
		receipt, found, err := runrecord.ResolveStageReceipt(context.Background(), fixture.store, execution.Operation, node.ID)
		if err != nil || !found || receipt.State != runrecord.StageCompleted {
			t.Fatalf("restart stage %s=(%+v, %t, %v)", node.ID, receipt, found, err)
		}
	}
}
