package processlock

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestAcquireContextContentionIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	ctx, stop := context.WithCancelCause(t.Context())
	before := errors.New("cancelled before acquisition")
	stop(before)
	if lock, err := AcquireContext(ctx, path, 0o600); !errors.Is(err, before) || errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("cancelled before acquisition = %v; want the caller's cause without contention", err)
	}
	owner, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	// A waiter the caller ends carries the caller's cause; the contention
	// itself is proven beside it by the owner's undisturbed lock.
	ended := errors.New("the caller ended the contended acquisition")
	contended, end := context.WithCancelCause(t.Context())
	type acquisition struct {
		lock *Lock
		err  error
	}
	waited := make(chan acquisition, 1)
	go func() { lock, err := AcquireContext(contended, path, 0o600); waited <- acquisition{lock, err} }()
	end(ended)
	if got := <-waited; !errors.Is(got.err, ended) {
		_ = got.lock.Close()
		t.Fatalf("cancelled contended acquisition = %v; want the caller's cause", got.err)
	}
	if lock, err := Acquire(path, 0o600); !errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("cancelled waiter disturbed owner: %v", err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	lock, err := AcquireContext(t.Context(), path, 0o600)
	if err != nil {
		t.Fatalf("released owner left a waiter behind: %v", err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAcquireIsExclusiveAndReleasedByClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	first, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Acquire(path, 0o600); !errors.Is(err, ErrBusy) {
		_ = second.Close()
		t.Fatalf("competing acquisition = %v; want OS-proven contention", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatalf("reacquire after close: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAcquireRejectsFalseContention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "process.lock")
	if err := os.WriteFile(path, []byte("stale owner metadata"), 0o600); err != nil {
		t.Fatal(err)
	}
	lock, err := Acquire(path, 0o600)
	if err != nil {
		t.Fatalf("unowned lock file blocked acquisition: %v", err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if lock, err := Acquire(t.TempDir(), 0o600); err == nil || errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("invalid lock path classified as contention: %v", err)
	}
}

func TestAcquireReleasedAfterProcessDeath(t *testing.T) {
	const helperPath = "OVERGO_TEST_LOCK_PROCESS_PATH"
	if path := os.Getenv(helperPath); path != "" {
		lock, err := Acquire(path, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		fmt.Println("locked")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	path := filepath.Join(t.TempDir(), "process.lock")
	child := exec.Command(os.Args[0], "-test.run=^TestAcquireReleasedAfterProcessDeath$", "-test.count=1")
	child.Env = append(os.Environ(), helperPath+"="+path)
	input, err := child.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	output, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	child.Stderr = os.Stderr
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = child.Process.Kill(); _ = child.Wait() })
	var ready string
	if _, err := fmt.Fscan(output, &ready); err != nil || ready != "locked" {
		t.Fatalf("child readiness = %q, %v", ready, err)
	}
	if lock, err := Acquire(path, 0o600); !errors.Is(err, ErrBusy) {
		_ = lock.Close()
		t.Fatalf("live writer contention = %v", err)
	}
	type acquisition struct {
		lock *Lock
		err  error
	}
	queued := make(chan acquisition, 1)
	// A waiter that returned before the owner's death holds an error the
	// receive below reports; the exclusive owner leaves it no early success.
	go func() { lock, err := AcquireContext(t.Context(), path, 0o600); queued <- acquisition{lock, err} }()
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = child.Wait()
	// The file remains. Only OS ownership determines whether reuse is safe.
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	got := <-queued
	if got.err != nil {
		t.Fatalf("owner death failed queued acquisition: %v", got.err)
	}
	if err := got.lock.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAcquireCancellationReleaseRace(t *testing.T) {
	for _, order := range []string{"cancel-first", "release-first", "concurrent"} {
		t.Run(order, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "process.lock")
			owner, err := Acquire(path, 0o600)
			if err != nil {
				t.Fatal(err)
			}
			defer owner.Close()
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			done := make(chan error, 1)
			go func() {
				lock, err := AcquireContext(ctx, path, 0o600)
				done <- errors.Join(err, lock.Close())
			}()
			switch order {
			case "cancel-first":
				cancel(context.Canceled)
				_ = owner.Close()
			case "release-first":
				_ = owner.Close()
				cancel(context.Canceled)
			default:
				cancelled := make(chan struct{})
				go func() { cancel(context.Canceled); close(cancelled) }()
				_ = owner.Close()
				<-cancelled
			}
			if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			lock, err := Acquire(path, 0o600)
			if err != nil {
				t.Fatalf("cancelled acquisition leaked ownership: %v", err)
			}
			if err := lock.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
