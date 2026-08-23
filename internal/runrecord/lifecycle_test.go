package runrecord

import (
	"testing"
	"time"

	"overgo/internal/artifact"
)

func TestGateLifecycleAndDebt(t *testing.T) {
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-environment"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-result"))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := NewGatePreparation(string(make([]byte, 64)), environment, time.Unix(100, 0))
	if err == nil {
		t.Fatal("accepted a non-hex tree key")
	}
	prepared, err = NewGatePreparation("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", environment, time.Unix(100, 0))
	if err != nil {
		t.Fatal(err)
	}
	debt, err := OutstandingGateDebt([]GateLifecycle{prepared})
	if err != nil || len(debt) != 1 || debt[0].ID != prepared.ID {
		t.Fatalf("prepared debt = (%+v, %v)", debt, err)
	}
	finalized, err := NewGateFinalization(prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded)
	if err != nil {
		t.Fatal(err)
	}
	debt, err = OutstandingGateDebt([]GateLifecycle{prepared, finalized})
	if err != nil || len(debt) != 0 {
		t.Fatalf("finalized debt = (%+v, %v)", debt, err)
	}
}

func TestGateHeartbeat(t *testing.T) {
	now := time.Unix(100, 0)
	heartbeat := GateHeartbeat{State: HeartbeatRunning, Updated: now}
	if got := heartbeat.Watchdog(now.Add(4*time.Second), 5*time.Second); got != HeartbeatRunning {
		t.Fatalf("fresh heartbeat = %s", got)
	}
	if got := heartbeat.Watchdog(now.Add(6*time.Second), 5*time.Second); got != HeartbeatStale {
		t.Fatalf("old heartbeat = %s", got)
	}
	heartbeat.State = HeartbeatRecordDebt
	if got := heartbeat.Watchdog(now.Add(time.Hour), 5*time.Second); got != HeartbeatRecordDebt {
		t.Fatalf("terminal heartbeat = %s", got)
	}
}
