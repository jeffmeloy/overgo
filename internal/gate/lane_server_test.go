package gate

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
)

// TestLaneServerEndsWithRun pins the lane runner: a server the lane leaves
// running ends when the lane exits, a lane that outlives its check is
// terminated with its server, and either way the resource the server
// claimed admits a fresh claim once the runner has returned; the receipt
// reports the exit. This test binary serves as the lane and as its server.
func TestLaneServerEndsWithRun(t *testing.T) {
	switch os.Getenv("OVERGO_GATE_LANE_HELPER") {
	case "server":
		if err := processcontrol.ClaimResource(os.Getenv("OVERGO_GATE_LANE_RESOURCE")); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(os.Getenv("OVERGO_GATE_LANE_MARKER"), []byte("claimed"), 0o600); err != nil {
			t.Fatal(err)
		}
		time.Sleep(2 * time.Minute)
		return
	case "lane":
		// The server's output goes nowhere: a real lane's server writes to
		// pipes the lane drains, never to the lane's own.
		server := exec.Command(os.Args[0], "-test.run=^TestLaneServerEndsWithRun$")
		server.Env = append(os.Environ(), "OVERGO_GATE_LANE_HELPER=server")
		if err := server.Start(); err != nil {
			t.Fatal(err)
		}
		awaitLaneMarker(os.Getenv("OVERGO_GATE_LANE_MARKER"))
		if os.Getenv("OVERGO_GATE_LANE_BLOCK") != "" {
			time.Sleep(2 * time.Minute)
		}
		return
	}
	for _, blocking := range []bool{false, true} {
		dir := t.TempDir()
		marker := filepath.Join(dir, "claimed")
		resource := "gate-lane-test-" + filepath.Base(dir)
		environment := append(os.Environ(), "OVERGO_GATE_LANE_HELPER=lane", "OVERGO_GATE_LANE_RESOURCE="+resource, "OVERGO_GATE_LANE_MARKER="+marker)
		ctx, cancel := context.WithCancelCause(t.Context())
		if blocking {
			environment = append(environment, "OVERGO_GATE_LANE_BLOCK=1")
			go func() {
				awaitLaneMarker(marker)
				cancel(errors.New("the check ended"))
			}()
		}
		receipt, err := superviseLane(ctx, environment, dir, os.Args[0], "-test.run=^TestLaneServerEndsWithRun$")
		cancel(nil)
		if blocking {
			if err == nil || !strings.Contains(err.Error(), "lane tree terminated") || !strings.Contains(err.Error(), "the check ended") {
				t.Fatalf("blocking lane: receipt %q, err %v; want the tree terminated by the check's end", receipt, err)
			}
		} else if err != nil || !strings.HasPrefix(receipt, "lane tree ended: exit=0 tree_terminated=false") {
			t.Fatalf("exiting lane: receipt %q, err %v", receipt, err)
		}
		// The server's claim is gone once the runner has returned.
		admitted, done := context.WithTimeoutCause(t.Context(), 10*time.Second, errors.New("the lane's server still holds its resource"))
		err = processcontrol.AwaitResource(admitted, func() error { return processcontrol.ClaimResource(resource) })
		done()
		if err != nil {
			t.Fatalf("blocking=%t: %v", blocking, err)
		}
	}
}

// awaitLaneMarker waits for the server's claim marker to appear.
func awaitLaneMarker(marker string) {
	for {
		if _, err := os.Stat(marker); err == nil {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}
