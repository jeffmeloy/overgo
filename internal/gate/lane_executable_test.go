package gate

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/atomicfile"
	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
)

func TestLaneExecutableLifetime(t *testing.T) {
	const addressKey = "OVERGO_LANE_EXECUTABLE_TEST_ADDRESS"
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if address := os.Getenv(addressKey); address != "" {
		connection, err := net.Dial("tcp", address)
		if err != nil {
			t.Fatal(err)
		}
		defer connection.Close()
		// A runner already using the retained image must not overwrite itself.
		path, err := laneExecutable(filepath.Dir(filepath.Dir(executable)), executable)
		if err != nil || path != executable {
			t.Fatalf("self reuse: %q, %v", path, err)
		}
		if _, err := fmt.Fprintln(connection, "ready"); err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(io.Discard, connection); err != nil {
			t.Fatal(err)
		}
		return
	}
	repo := t.TempDir()
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(executable)
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(repo, filepath.Base(executable))
	if err := atomicfile.Write(source, data, info.Mode().Perm()); err != nil {
		t.Fatal(err)
	}
	retained, err := laneExecutable(repo, source)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(retained)
	if err != nil || !bytes.Equal(data, got) {
		t.Fatalf("image identity: %v", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	stop := context.AfterFunc(t.Context(), func() { _ = listener.Close() })
	defer stop()
	log, err := os.Create(filepath.Join(repo, "child.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	pid, err := processcontrol.StartDetached(processcontrol.Command{
		Path: retained, Args: []string{"-test.run=^TestLaneExecutableLifetime$"},
		Env: append(os.Environ(), addressKey+"="+listener.Addr().String()),
	}, log)
	if err != nil {
		t.Fatal(err)
	}
	child, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = child.Kill(); _, _ = child.Wait() }()
	connection, err := listener.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	var ready string
	if _, err := fmt.Fscanln(connection, &ready); err != nil || ready != "ready" {
		t.Fatalf("child readiness: %q %v", ready, err)
	}
	// On Windows this fails if the detached child still maps the temporary image.
	if err := os.Remove(source); err != nil {
		t.Fatalf("temporary image remains loaded: %v", err)
	}
	if !processcontrol.ProcessAlive(pid) {
		t.Fatal("child exited with its temporary source")
	}
	if err := connection.Close(); err != nil {
		t.Fatal(err)
	}
	state, err := child.Wait()
	if err != nil || !state.Success() {
		t.Fatalf("detached exit: %v %v", state, err)
	}
	// The next admitted runner replaces the same slot, with no executable history.
	if err := os.WriteFile(source, []byte("replacement"), clioptions.OutputFileMode); err != nil {
		t.Fatal(err)
	}
	next, err := laneExecutable(repo, source)
	if err != nil || next != retained {
		t.Fatalf("replacement: %q %v", next, err)
	}
	got, err = os.ReadFile(next)
	if err != nil || string(got) != "replacement" {
		t.Fatalf("replacement bytes: %q %v", got, err)
	}
	if _, err := laneExecutable(repo, filepath.Join(repo, "absent")); err == nil {
		t.Fatal("missing executable accepted")
	}
	if err := os.Remove(retained); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(retained, clioptions.OutputDirectoryMode); err != nil {
		t.Fatal(err)
	}
	if _, err := laneExecutable(repo, source); err == nil {
		t.Fatal("failed replacement accepted")
	}
}
