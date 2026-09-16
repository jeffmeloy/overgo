//go:build windows

package processcontrol

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestDetachedJobChildHelper(t *testing.T) {
	address := os.Getenv("OVERGO_DETACHED_CHILD_ADDRESS")
	if address == "" {
		return
	}
	if os.Getenv("OVERGO_DETACHED_CHILD_ROLE") == "parent" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		child := exec.Command(executable, "-test.run=^TestDetachedJobChildHelper$")
		child.Env = append(os.Environ(), "OVERGO_DETACHED_CHILD_ROLE=child")
		if os.Getenv("OVERGO_DETACHED_CHILD_PIPE") == "1" {
			child.Stdout = os.Stdout
		}
		ready, err := child.StderrPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := child.Start(); err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadFull(ready, make([]byte, len("ready"))); err != nil {
			t.Fatal(err)
		}
		_ = ready.Close()
		if os.Getenv("OVERGO_DETACHED_CHILD_BLOCK") == "1" {
			if err := child.Wait(); err != nil {
				t.Fatal(err)
			}
		} else if err := child.Process.Release(); err != nil {
			t.Fatal(err)
		}
		return
	}
	connection, err := net.Dial("tcp", address)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	if err := binary.Write(connection, binary.LittleEndian, uint32(os.Getpid())); err != nil {
		t.Fatal(err)
	}
	// The test retains our process handle before allowing the parent to exit.
	if _, err := io.ReadFull(connection, make([]byte, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stderr.Write([]byte("ready")); err != nil {
		t.Fatal(err)
	}
	_ = os.Stderr.Close()
	if _, err := io.Copy(io.Discard, connection); err != nil {
		t.Fatal(err)
	}
}

func TestWindowsWaitIncludesDetachedDescendant(t *testing.T) {
	for _, testCase := range []struct {
		name            string
		cancel, inherit bool
	}{
		{"parent_exit", false, false}, {"parent_exit_inherited_pipe", false, true},
		{"canceled", true, false}, {"canceled_inherited_pipe", true, true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			stopAccept := context.AfterFunc(t.Context(), func() { _ = listener.Close() })
			defer stopAccept()
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			environment := append(os.Environ(), "OVERGO_DETACHED_CHILD_ADDRESS="+listener.Addr().String(), "OVERGO_DETACHED_CHILD_ROLE=parent")
			if testCase.inherit {
				environment = append(environment, "OVERGO_DETACHED_CHILD_PIPE=1")
			}
			if testCase.cancel {
				environment = append(environment, "OVERGO_DETACHED_CHILD_BLOCK=1")
			}
			supervised, err := Start(t.Context(), Command{Path: executable, Args: []string{"-test.run=^TestDetachedJobChildHelper$"}, Env: environment})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if !supervised.Exited() {
					_ = supervised.Terminate()
				}
				_, _ = supervised.Wait(context.WithoutCancel(t.Context()))
			}()
			connection, err := listener.Accept()
			if err != nil {
				t.Fatal(err)
			}
			defer connection.Close()
			var pid uint32
			if err := binary.Read(connection, binary.LittleEndian, &pid); err != nil {
				t.Fatal(err)
			}
			handle, err := syscall.OpenProcess(syscall.SYNCHRONIZE, false, pid)
			if err != nil {
				t.Fatal(err)
			}
			defer syscall.CloseHandle(handle)
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			// The wait ends with the process or job event the release publishes;
			// a missing trigger hangs here and the test binary reports it.
			waitContext := ctx
			if _, err := connection.Write([]byte{1}); err != nil {
				t.Fatal(err)
			}
			if testCase.cancel {
				cancel(context.Canceled)
			}
			receipt, err := supervised.Wait(waitContext)
			if receipt.TreeTerminated != testCase.cancel || (err != nil) != testCase.cancel {
				t.Fatalf("terminal receipt: %+v, %v", receipt, err)
			}
			state, err := syscall.WaitForSingleObject(handle, 0)
			if err != nil || state != syscall.WAIT_OBJECT_0 {
				t.Fatalf("receipt preceded child exit: %d, %v", state, err)
			}
			if !supervised.Exited() {
				t.Fatal("terminal wait did not publish exit")
			}
		})
	}
}
