package device

import (
	"errors"
	"sync"
	"testing"
)

func TestWorkerCloseMemoizesShutdownError(t *testing.T) {
	want := errors.New("shutdown failed")
	worker := &Worker{
		stop: make(chan chan error),
		done: make(chan struct{}),
	}
	worker.close = sync.OnceValue(worker.stopAndWait)
	go func() {
		result := <-worker.stop
		result <- want
	}()

	if got := worker.Close(); !errors.Is(got, want) {
		t.Fatalf("first Close() error = %v, want %v", got, want)
	}
	if got := worker.Close(); !errors.Is(got, want) {
		t.Fatalf("second Close() error = %v, want memoized %v", got, want)
	}
}
