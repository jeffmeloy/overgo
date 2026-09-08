//go:build windows

package processcontrol

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

const sharedAdmissionProbeEnvironment = "OVERGO_TEST_SHARED_RESOURCE"

type sharedAdmissionProbe struct {
	Name                                           string
	Exclusive, Hold, Busy, Descendant, Transaction bool
}

func TestGPUCapacityAdmissionProcess(t *testing.T) {
	encoded := os.Getenv(sharedAdmissionProbeEnvironment)
	if encoded == "" {
		return
	}
	var probe sharedAdmissionProbe
	if err := json.Unmarshal([]byte(encoded), &probe); err != nil {
		t.Fatal(err)
	}
	if probe.Transaction {
		if err := ResourceTransaction(probe.Name, func() error {
			fmt.Println("admitted")
			_, err := io.Copy(io.Discard, os.Stdin)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return
	}
	var release func() error
	var err error
	if probe.Exclusive {
		err = ClaimResource(probe.Name)
		release = func() error { return nil }
	} else {
		release, err = ShareResource(probe.Name)
	}
	if probe.Busy {
		if !errors.Is(err, ErrResourceBusy) {
			t.Fatalf("wanted typed contention, got %v", err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if probe.Descendant {
		child := sharedAdmissionCommand(t, sharedAdmissionProbe{Name: probe.Name, Hold: true})
		child.Stdin = os.Stdin
		child.Stderr = os.Stderr
		output, err := child.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := child.Start(); err != nil {
			t.Fatal(err)
		}
		line, err := bufio.NewReader(output).ReadString('\n')
		if err != nil || strings.TrimSpace(line) != "admitted" {
			t.Fatalf("native descendant admission: %q: %v", line, err)
		}
		fmt.Printf("descendant %d\nadmitted\n", child.Process.Pid)
		_, _ = io.Copy(io.Discard, output)
		if err := child.Wait(); err != nil {
			t.Fatal(err)
		}
		return
	}
	if !probe.Hold {
		return
	}
	fmt.Println("admitted")
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		if input.Text() != "release" {
			t.Fatalf("unknown probe control %q", input.Text())
		}
		if err := errors.Join(release(), release()); err != nil {
			t.Fatal(err)
		}
		fmt.Println("released")
	}
	if err := input.Err(); err != nil {
		t.Fatal(err)
	}
}

func sharedAdmissionCommand(t *testing.T, probe sharedAdmissionProbe) *exec.Cmd {
	t.Helper()
	encoded, err := json.Marshal(probe)
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestGPUCapacityAdmissionProcess$", "-test.timeout=30s")
	command.Dir = t.TempDir()
	command.Env = append(os.Environ(), sharedAdmissionProbeEnvironment+"="+string(encoded), "OVERGO_DATA_ROOT="+command.Dir, "ProgramData="+command.Dir)
	return command
}

func assertSharedAdmission(t *testing.T, name string, exclusive, busy bool) {
	t.Helper()
	command := sharedAdmissionCommand(t, sharedAdmissionProbe{Name: name, Exclusive: exclusive, Busy: busy})
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("exclusive=%t busy=%t: %v: %s", exclusive, busy, err, output)
	}
}

type heldSharedAdmission struct {
	command    *exec.Cmd
	input      io.WriteCloser
	output     *bufio.Reader
	descendant *os.Process
}

func holdSharedAdmission(t *testing.T, probe sharedAdmissionProbe) heldSharedAdmission {
	t.Helper()
	probe.Hold = true
	command := sharedAdmissionCommand(t, probe)
	reader, input, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	command.Stdin = reader
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = input.Close(); _ = command.Process.Kill(); _ = command.Wait() })
	held := heldSharedAdmission{command: command, input: input, output: bufio.NewReader(output)}
	if probe.Descendant {
		line, err := held.output.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		encodedPID, found := strings.CutPrefix(strings.TrimSpace(line), "descendant ")
		pid, err := strconv.Atoi(encodedPID)
		if !found || err != nil {
			t.Fatalf("descendant identity: %q: %v", line, err)
		}
		held.descendant, err = os.FindProcess(pid)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = held.descendant.Kill(); _, _ = held.descendant.Wait() })
	}
	held.requireLine(t, "admitted")
	return held
}

func (held heldSharedAdmission) requireLine(t *testing.T, want string) {
	t.Helper()
	line, err := held.output.ReadString('\n')
	if err != nil || strings.TrimSpace(line) != want {
		t.Fatalf("resource barrier: got %q, want %q: %v", line, want, err)
	}
}

func (held heldSharedAdmission) release(t *testing.T) {
	t.Helper()
	if _, err := io.WriteString(held.input, "release\n"); err != nil {
		t.Fatal(err)
	}
	held.requireLine(t, "released")
}

func (held heldSharedAdmission) kill(t *testing.T) {
	t.Helper()
	if err := held.command.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := held.command.Wait(); err == nil {
		t.Fatal("owner death was not injected")
	}
}

func sharedAdmissionName(t *testing.T) string {
	t.Helper()
	name := "gpu-admission:" + t.TempDir()
	path, err := resourceAdmissionPath(name)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(path) })
	return name
}

func TestGPUCapacityAdmissionConcurrent(t *testing.T) {
	name := sharedAdmissionName(t)
	one := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	two := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	assertSharedAdmission(t, name, false, false)
	assertSharedAdmission(t, name, true, true)
	assertSharedAdmission(t, sharedAdmissionName(t), true, false)
	one.release(t)
	assertSharedAdmission(t, name, true, true)
	two.release(t)
	assertSharedAdmission(t, name, true, false)
	t.Log("two independent readers; distinct worktrees/data roots; each release required; idle live processes do not retain admission")
}

func TestGPUCapacityAdmissionProcessExit(t *testing.T) {
	name := sharedAdmissionName(t)
	one := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	two := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	one.kill(t)
	assertSharedAdmission(t, name, true, true)
	two.kill(t)
	assertSharedAdmission(t, name, true, false)
	owner := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true})
	assertSharedAdmission(t, name, false, true)
	owner.kill(t)
	assertSharedAdmission(t, name, false, false)
	t.Log("shared and exclusive crash cleanup; stale empty file grants no ownership")
}

func TestGPUExclusiveMeasurementAdmission(t *testing.T) {
	name := sharedAdmissionName(t)
	owner := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true, Descendant: true})
	assertSharedAdmission(t, name, true, true)
	assertSharedAdmission(t, name, false, true)
	owner.kill(t)
	assertSharedAdmission(t, name, false, true)
	assertSharedAdmission(t, name, true, true)
	if err := owner.input.Close(); err != nil {
		t.Fatal(err)
	}
	if state, err := owner.descendant.Wait(); err != nil || !state.Success() {
		t.Fatalf("native descendant exit: %v: %v", state, err)
	}
	reader := holdSharedAdmission(t, sharedAdmissionProbe{Name: name})
	assertSharedAdmission(t, name, true, true)
	reader.release(t)
	assertSharedAdmission(t, name, true, false)
}
