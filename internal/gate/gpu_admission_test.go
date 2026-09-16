package gate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
)

const resourceProcessEnvironment = "OVERGO_TEST_RESOURCE_PROCESS"

type resourceProcessSpec struct {
	Name, Role, Root string
	// Announce is the address the process reports its claim to and holds
	// its claim against: the connection's close ends the process.
	Announce string
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
		// The owner's children announce themselves to the test, which holds
		// their claims apart from the owner's life.
		for _, role := range []string{"child", "sibling"} {
			child := resourceProcessCommand(t, resourceProcessSpec{Name: spec.Name, Role: role, Root: spec.Root, Announce: spec.Announce})
			child.Dir = spec.Root
			if err := child.Start(); err != nil {
				t.Fatal(err)
			}
			defer func() { _ = child.Process.Kill(); _ = child.Wait() }()
		}
	}
	// The process announces its claim over the connection and holds it
	// until the test closes that connection.
	connection, err := net.Dial("tcp", spec.Announce)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(connection, "%s %d\n", spec.Role, os.Getpid())
	_, _ = io.Copy(io.Discard, connection)
}

// resourceProcesses accepts the announcements of resource processes and
// holds each claim open until the test releases it.
type resourceProcesses struct {
	t        *testing.T
	listener net.Listener
	held     map[string]heldResource
}

type heldResource struct {
	pid        int
	connection net.Conn
}

func listenResourceProcesses(t *testing.T) *resourceProcesses {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	return &resourceProcesses{t: t, listener: listener, held: map[string]heldResource{}}
}

func (r *resourceProcesses) address() string { return r.listener.Addr().String() }

// ready accepts announcements until the role's arrives, then reports its pid.
func (r *resourceProcesses) ready(role string) int {
	r.t.Helper()
	for {
		if held, announced := r.held[role]; announced {
			return held.pid
		}
		connection, err := r.listener.Accept()
		if err != nil {
			r.t.Fatal(err)
		}
		line, err := bufio.NewReader(connection).ReadString('\n')
		if err != nil {
			r.t.Fatal(err)
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			r.t.Fatalf("resource announcement %q", line)
		}
		pid, err := strconv.Atoi(fields[1])
		if err != nil {
			r.t.Fatal(err)
		}
		r.held[fields[0]] = heldResource{pid: pid, connection: connection}
		r.t.Cleanup(func() { _ = connection.Close() })
	}
}

// release closes the role's connection, ending its process.
func (r *resourceProcesses) release(role string) {
	r.t.Helper()
	held, announced := r.held[role]
	if !announced {
		r.t.Fatalf("release of %s before its announcement", role)
	}
	if err := held.connection.Close(); err != nil {
		r.t.Fatal(err)
	}
}

// waitProcessExit returns once the process has exited, through its own
// process object.
func waitProcessExit(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return nil
	}
	_, err = process.Wait()
	return err
}

// startResourceProcess starts one resource process announcing to the test's
// listener, with its output copied to output when set.
func startResourceProcess(t *testing.T, processes *resourceProcesses, spec resourceProcessSpec, output io.Writer) *exec.Cmd {
	t.Helper()
	spec.Announce = processes.address()
	command := resourceProcessCommand(t, spec)
	command.Dir = spec.Root
	command.Stdout, command.Stderr = output, output
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	return command
}

func TestCrossWorktreeGPUAdmission(t *testing.T) {
	t.Parallel()
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
	var output bytes.Buffer
	processes := listenResourceProcesses(t)
	owner := startResourceProcess(t, processes, resourceProcessSpec{Name: name, Role: "owner", Root: root}, &output)
	t.Cleanup(func() { _ = owner.Process.Kill() })
	processes.ready("owner")
	processes.ready("child")
	sibling := processes.ready("sibling")
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
	processes.release("child")
	assertBlocked()
	// The last consumer's exit releases the reservation; its process object
	// signals the exit.
	processes.release("sibling")
	if err := waitProcessExit(sibling); err != nil {
		t.Fatal(err)
	}
	if err := probe(name, other); err != nil {
		t.Fatalf("reservation survived the last consumer: %v; owner: %s", err, output.String())
	}
	t.Log("OS admission: separate worktrees, stale heartbeat, reentry, launcher death with live consumer, final release, independent device and CPU")
}
