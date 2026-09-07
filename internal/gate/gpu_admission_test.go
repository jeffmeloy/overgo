package gate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
)

const resourceProcessEnvironment = "OVERGO_TEST_RESOURCE_PROCESS"

type resourceProcessSpec struct {
	Name, Role, Root string
}

func resourceProcessCommand(t *testing.T, spec resourceProcessSpec) *exec.Cmd {
	t.Helper()
	payload, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestGPUAdmissionProcess$", "-test.timeout=30s")
	command.Env = append(os.Environ(), resourceProcessEnvironment+"="+string(payload))
	return command
}

func TestGPUAdmissionProcess(t *testing.T) {
	value := os.Getenv(resourceProcessEnvironment)
	if value == "" {
		return
	}
	var spec resourceProcessSpec
	if err := json.Unmarshal([]byte(value), &spec); err != nil {
		t.Fatal(err)
	}
	if err := processcontrol.ClaimResource(spec.Name); err != nil {
		t.Fatal(err)
	}
	if err := processcontrol.ClaimResource(spec.Name); err != nil {
		t.Fatalf("same-process reentry: %v", err)
	}
	if spec.Role == "probe" {
		return
	}
	if spec.Role == "owner" {
		for _, role := range []string{"child", "sibling"} {
			child := resourceProcessCommand(t, resourceProcessSpec{Name: spec.Name, Role: role, Root: spec.Root})
			if err := child.Start(); err != nil {
				t.Fatal(err)
			}
			defer func() { _ = child.Process.Kill(); _ = child.Wait() }()
			waitResourceFile(t, filepath.Join(spec.Root, role+"-ready"))
		}
	}
	if err := os.WriteFile(filepath.Join(spec.Root, spec.Role+"-ready"), []byte(fmt.Sprint(os.Getpid())), 0600); err != nil {
		t.Fatal(err)
	}
	waitResourceFile(t, filepath.Join(spec.Root, spec.Role+"-stop"))
}

func waitResourceFile(t *testing.T, path string) {
	t.Helper()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 20*time.Second, errors.New("resource process did not publish its state"))
	defer cancel()
	ticker := time.Tick(10 * time.Millisecond)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("resource process did not publish %s: %v", path, ctx.Err())
		case <-ticker:
		}
	}
}

func TestCrossWorktreeGPUAdmission(t *testing.T) {
	root := t.TempDir()
	name := "gpu-test:" + root
	probe := func(resource, directory string) error {
		t.Helper()
		command := resourceProcessCommand(t, resourceProcessSpec{Name: resource, Role: "probe", Root: directory})
		command.Dir = directory
		output, err := command.CombinedOutput()
		if err != nil {
			return fmt.Errorf("%w: %s", err, output)
		}
		return nil
	}
	if runtime.GOOS != "windows" {
		if err := probe(name, root); err == nil || !strings.Contains(err.Error(), "requires Windows job objects") {
			t.Fatalf("unsupported isolation must refuse: %v", err)
		}
		return
	}
	other := t.TempDir()
	owner := resourceProcessCommand(t, resourceProcessSpec{Name: name, Role: "owner", Root: root})
	owner.Dir = root
	var output bytes.Buffer
	owner.Stdout, owner.Stderr = &output, &output
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = owner.Process.Kill()
		_ = os.WriteFile(filepath.Join(root, "child-stop"), nil, 0600)
		_ = os.WriteFile(filepath.Join(root, "sibling-stop"), nil, 0600)
	})
	waitResourceFile(t, filepath.Join(root, "owner-ready"))
	if err := os.WriteFile(filepath.Join(root, "expired-heartbeat"), []byte("1970-01-01"), 0600); err != nil {
		t.Fatal(err)
	}
	assertBlocked := func() {
		t.Helper()
		if err := probe(name, other); err == nil || !strings.Contains(err.Error(), "already reserved") {
			t.Fatalf("another worktree acquired an occupied physical resource: %v", err)
		}
	}
	assertBlocked()
	if err := probe(name+":other-device", other); err != nil {
		t.Fatalf("independent device blocked: %v", err)
	}
	cpu := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestGPUAdmissionProcess$")
	if result, err := cpu.CombinedOutput(); err != nil {
		t.Fatalf("independent CPU work blocked: %v: %s", err, result)
	}
	if err := owner.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := owner.Wait(); err == nil {
		t.Fatal("launcher death was not injected")
	}
	assertBlocked()
	if err := os.WriteFile(filepath.Join(root, "child-stop"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	assertBlocked()
	if err := os.WriteFile(filepath.Join(root, "sibling-stop"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		if err := probe(name, other); err == nil {
			break
		} else if time.Now().After(deadline) {
			t.Fatalf("reservation survived the last consumer: %v; owner: %s", err, output.String())
		}
	}
	t.Log("OS admission: separate worktrees, stale heartbeat, reentry, launcher death with live consumer, final release, independent device and CPU")
}
