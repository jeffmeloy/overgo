//go:build windows

package gate

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
)

func TestHostTestResourceAdmission(t *testing.T) {
	root := t.TempDir()
	name := "test-batch:" + root
	owner := resourceProcessCommand(t, resourceProcessSpec{Name: name, Role: "holder", Root: root})
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = owner.Process.Kill() })
	waitResourceFile(t, filepath.Join(root, "holder-ready"))
	ctx, cancel := context.WithTimeoutCause(t.Context(), 100*time.Millisecond, context.DeadlineExceeded)
	defer cancel()
	release, err := admitSharedTestResource(ctx, name)
	if release != nil || !errors.Is(err, processcontrol.ErrResourceBusy) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("exclusive owner was not respected within deadline: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "holder-stop"), nil, 0600); err != nil {
		t.Fatal(err)
	}
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
