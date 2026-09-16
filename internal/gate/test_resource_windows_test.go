//go:build windows

package gate

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
)

func TestHostTestResourceAdmission(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	name := "test-batch:" + root
	processes := listenResourceProcesses(t)
	owner := startResourceProcess(t, processes, resourceProcessSpec{Name: name, Role: "holder", Root: root}, nil)
	t.Cleanup(func() { _ = owner.Process.Kill() })
	processes.ready("holder")
	// An admission whose caller has left waits for no holder and carries
	// the caller's cause; the holder's claim stands beside it.
	left := errors.New("the admission's caller left")
	ctx, leave := context.WithCancelCause(t.Context())
	leave(left)
	release, err := admitSharedTestResource(ctx, name)
	if release != nil || !errors.Is(err, left) {
		t.Fatalf("exclusive owner was not respected: %v", err)
	}
	processes.release("holder")
	if err := owner.Wait(); err != nil {
		t.Fatal(err)
	}
	release, err = admitSharedTestResource(t.Context(), name)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	other, err := processcontrol.ShareResource(name)
	if err != nil {
		t.Fatalf("correctness batch prevented another shared consumer: %v", err)
	}
	defer other()
	probe := resourceProcessCommand(t, resourceProcessSpec{Name: name, Role: "probe", Root: root})
	if out, err := probe.CombinedOutput(); err == nil || !strings.Contains(string(out), "already reserved") {
		t.Fatalf("measurement entered shared batch: %v: %s", err, out)
	}
	if err := errors.Join(release(), other()); err != nil {
		t.Fatal(err)
	}
	probe = resourceProcessCommand(t, resourceProcessSpec{Name: name, Role: "probe", Root: root})
	if out, err := probe.CombinedOutput(); err != nil {
		t.Fatalf("batch release retained reservation: %v: %s", err, out)
	}
}
