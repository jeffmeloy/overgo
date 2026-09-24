//go:build windows

package fsatomic

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

const holderEnvironment = "OVERGO_FSATOMIC_HOLDER"

// mapForReading maps path read-only through a handle that shares every
// access, the way a background git status maps a worktree file it hashes,
// and returns the release.
func mapForReading(t *testing.T, path string) func() {
	t.Helper()
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := syscall.CreateFileMapping(handle, nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	view, err := syscall.MapViewOfFile(mapping, syscall.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return func() {
		_ = syscall.UnmapViewOfFile(view)
		_ = syscall.CloseHandle(mapping)
		_ = syscall.CloseHandle(handle)
	}
}

// TestReplaceProceedsWhenTheHolderReleasesAndStaysAlive holds the replace to
// the release of the file, never the holder's lifetime. A holder mapping the
// destination with full sharing -- a background git status -- does not block
// the replace at all. A holder that denies deletion is named, and once it
// closes the file while staying alive the next replace goes through. A holder
// in this process is named the same way.
func TestReplaceProceedsWhenTheHolderReleasesAndStaysAlive(t *testing.T) {
	if target := os.Getenv(holderEnvironment); target != "" {
		input := bufio.NewReader(os.Stdin)
		if mode, _ := input.ReadString('\n'); strings.TrimSpace(mode) == "shared" {
			release := mapForReading(t, target)
			fmt.Println("holding")
			_, _ = input.ReadString('\n')
			release()
		} else {
			file, err := os.Open(target) // no delete sharing
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("holding")
			_, _ = input.ReadString('\n')
			_ = file.Close()
		}
		fmt.Println("released")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	directory := t.TempDir()
	target := filepath.Join(directory, "compatibility.json")
	if err := os.WriteFile(target, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	replaceWith := func(content string) error {
		source := filepath.Join(directory, "next.json")
		if err := os.WriteFile(source, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return Replace(source, target)
	}
	requireTarget := func(want string) {
		t.Helper()
		if data, err := os.ReadFile(target); err != nil || string(data) != want {
			t.Fatalf("destination = %q %v, want %q", data, err, want)
		}
	}
	for _, mode := range []string{"shared", "denying"} {
		t.Run(mode+" holder", func(t *testing.T) {
			child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestReplaceProceedsWhenTheHolderReleasesAndStaysAlive$")
			child.Env = append(os.Environ(), holderEnvironment+"="+target)
			stdin, err := child.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			stdout, err := child.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := child.Start(); err != nil {
				t.Fatal(err)
			}
			defer func() { _ = stdin.Close(); _ = child.Wait() }()
			lines := bufio.NewReader(stdout)
			expect := func(want string) {
				t.Helper()
				if line, err := lines.ReadString('\n'); err != nil || strings.TrimSpace(line) != want {
					t.Fatalf("holder said %q %v, want %q", line, err, want)
				}
			}
			if _, err := fmt.Fprintln(stdin, mode); err != nil {
				t.Fatal(err)
			}
			expect("holding")
			if mode == "shared" {
				if err := replaceWith("shared " + mode); err != nil {
					t.Fatalf("a delete-sharing holder blocked the replace: %v", err)
				}
				requireTarget("shared " + mode)
				return
			}
			err = replaceWith("while held")
			if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("process %d", child.Process.Pid)) {
				t.Fatalf("a holder denying deletion was not named: %v", err)
			}
			if _, err := fmt.Fprintln(stdin, "release"); err != nil {
				t.Fatal(err)
			}
			expect("released")
			// The holder is still running: its release alone must be enough.
			if err := replaceWith("after release"); err != nil {
				t.Fatalf("the replace failed after the living holder released the file: %v", err)
			}
			requireTarget("after release")
		})
	}
	t.Run("this process", func(t *testing.T) {
		held, err := os.Open(target)
		if err != nil {
			t.Fatal(err)
		}
		defer held.Close()
		if err := replaceWith("again"); err == nil || !strings.Contains(err.Error(), fmt.Sprintf("process %d", os.Getpid())) {
			t.Fatalf("a holder in this process was not named: %v", err)
		}
	})
}
