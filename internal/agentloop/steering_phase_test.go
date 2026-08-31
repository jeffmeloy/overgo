package agentloop

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestToolCallPropensityPreservesEffectAuthority pins the separation between
// the intervention and the authority ladder: a steered phase can only shrink
// the visible action space and zero its own alpha -- the coordinator's
// mutation path still demands inspection, a preview-named grant, and the
// receipted execution, unchanged by any phase, and the phase type holds no
// handle that could grant, approve, dispatch, or receipt anything.
func TestToolCallPropensityPreservesEffectAuthority(t *testing.T) {
	ctx := t.Context()
	coordinator, _ := coordinatorFixture(t)
	direction := testutil.ArtifactID(t, artifact.KindRecipe, "propensity direction")
	manual := func(name string, effect agenttool.Effect) agenttool.Manual {
		value, err := agenttool.NewManual(agenttool.Manual{
			Name: name, Description: "Effect authority fixture.", Effect: effect,
			Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
		})
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	read := manual("probe.read", agenttool.EffectInspection)
	write := manual("probe.write", agenttool.EffectMutation)

	steered := ProposalSteeringPhase(direction, 0.6, []agenttool.Manual{read}, nil)
	if steered.Alpha != 0.6 || len(steered.Manuals) != 1 || steered.Manuals[0].Name != read.Name {
		t.Fatalf("inspection-only phase = %+v", steered)
	}
	exposed := ProposalSteeringPhase(direction, 0.6, []agenttool.Manual{read, write}, nil)
	if exposed.Alpha != 0 {
		t.Fatalf("mutation-exposing phase alpha = %v, want 0", exposed.Alpha)
	}
	for _, phaseManual := range exposed.Manuals {
		if phaseManual.Effect != agenttool.EffectInspection {
			t.Fatalf("steered phase exposes %s manual %q", phaseManual.Effect, phaseManual.Name)
		}
	}

	// The unsteered mutation ladder is byte-for-byte the same ceremony
	// whether or not a phase exists: inspect, preview, grant by the
	// previewed operation identity, then execute.
	session := &Session{ID: "effect-authority-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`), direction); err == nil ||
		!strings.Contains(err.Error(), "does not name the previewed operation") {
		t.Fatalf("phase direction accepted as a grant identity: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`),
		previewedOperation(t, coordinator, session, "probe.write")); err != nil {
		t.Fatal(err)
	}
	result, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`))
	if err != nil || string(result) != `{"changed":true}` {
		t.Fatalf("approved mutation = %s, %v", result, err)
	}
}
