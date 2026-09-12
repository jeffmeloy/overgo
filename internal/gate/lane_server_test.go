package gate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
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
		if os.Getenv("OVERGO_GATE_LANE_SERVER_FAIL") != "" {
			t.Fatal("intentional server refusal")
		}
		if err := processcontrol.ClaimResource(os.Getenv("OVERGO_GATE_LANE_RESOURCE")); err != nil {
			t.Fatal(err)
		}
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer listener.Close()
		fmt.Fprintln(os.Stdout, "ready")
		// Stay alive until the supervisor ends the tree. No timer can turn a
		// leaked server into an apparent successful cleanup.
		connection, err := listener.Accept()
		if connection != nil {
			connection.Close()
		}
		t.Fatalf("server outlived its expected termination: %v", err)
		return
	case "lane":
		// EOF reports failed readiness; an exited child cannot strand the lane.
		server := exec.Command(os.Args[0], "-test.run=^TestLaneServerEndsWithRun$")
		server.Env = append(os.Environ(), "OVERGO_GATE_LANE_HELPER=server")
		server.Stderr = os.Stderr
		output, err := server.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		defer output.Close()
		if err := server.Start(); err != nil {
			t.Fatal(err)
		}
		reader := bufio.NewReader(output)
		line, readErr := reader.ReadString('\n')
		if readErr != nil || line != "ready\n" {
			_ = server.Process.Kill()
			tail, _ := io.ReadAll(reader)
			t.Fatalf("server exited before readiness: output=%q read=%v exit=%v", line+string(tail), readErr, server.Wait())
		}
		connection, err := net.Dial("tcp", os.Getenv("OVERGO_GATE_LANE_READY"))
		if err != nil {
			t.Fatal(err)
		}
		ack, err := io.ReadAll(connection)
		connection.Close()
		if err != nil || string(ack) != "ready" {
			t.Fatalf("readiness acknowledgement: %q, %v", ack, err)
		}
		if os.Getenv("OVERGO_GATE_LANE_BLOCK") != "" {
			t.Fatalf("server ended before lane cancellation: %v", server.Wait())
		}
		return
	}
	t.Parallel()
	for _, tc := range []struct {
		name              string
		blocking, refusal bool
	}{{name: "normal"}, {name: "cancelled", blocking: true}, {name: "refused readiness", refusal: true}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			// Process and temporary directory isolate concurrent fixture claims.
			resource := fmt.Sprintf("gate-lane-test-%d-%s", os.Getpid(), dir)
			environment := append(os.Environ(), "OVERGO_GATE_LANE_HELPER=lane", "OVERGO_GATE_LANE_RESOURCE="+resource, "OVERGO_GATE_LANE_READY="+listener.Addr().String())
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			if tc.blocking {
				environment = append(environment, "OVERGO_GATE_LANE_BLOCK=1")
			}
			if tc.refusal {
				environment = append(environment, "OVERGO_GATE_LANE_SERVER_FAIL=1")
			}
			ready := make(chan error, 1)
			go func() {
				connection, err := listener.Accept()
				if err == nil {
					_, err = io.WriteString(connection, "ready")
					connection.Close()
					if tc.blocking {
						cancel(errors.New("the check ended"))
					}
				}
				ready <- err
			}()
			receipt, err := superviseLane(ctx, environment, dir, os.Args[0], "-test.run=^TestLaneServerEndsWithRun$")
			listener.Close()
			readyErr := <-ready
			cancel(nil)
			if tc.refusal {
				if err == nil || !strings.Contains(err.Error(), "server exited before readiness") {
					t.Fatalf("readiness failure lost: receipt %q, err %v", receipt, err)
				}
			} else if readyErr != nil {
				t.Fatalf("readiness: %v", readyErr)
			} else if tc.blocking {
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
				t.Fatalf("%s: %v", tc.name, err)
			}
		})
	}
}
