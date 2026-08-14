package main

import (
	"os"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/gatecontrol"
	"overgo/internal/runrecord"
)

func TestGateCommandLifecycleAdapter(t *testing.T) {
	control, err := gatecontrol.New(t.TempDir(), "store")
	if err != nil {
		t.Fatal(err)
	}
	preparation, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("preparation"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	g := gateContext{
		control: control,
		preparation: runrecord.GateLifecycle{
			ID: preparation, TreeKey: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		environment: runrecord.Environment{ID: environment},
	}
	heartbeat := g.heartbeat(runrecord.HeartbeatRunning)
	if heartbeat.Preparation != preparation || heartbeat.Environment != environment || heartbeat.PID != os.Getpid() || heartbeat.Updated.IsZero() {
		t.Fatalf("heartbeat adapter omitted gate facts: %+v", heartbeat)
	}
	if err := g.control.WriteHeartbeat(heartbeat); err != nil {
		t.Fatal(err)
	}
	status, err := g.control.Watchdog(time.Now().UTC(), time.Minute)
	if err != nil || status.State != runrecord.HeartbeatRunning {
		t.Fatalf("watchdog adapter = (%+v, %v)", status, err)
	}
}
