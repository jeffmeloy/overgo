//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

// TestFailedLoadEndsItsPromptPass holds the encoding of a request's prompts,
// begun beside the loading of the model that serves it, to whoever began it:
// nothing lets go of the pass while it runs.
//
// A load that fails returns what it failed with, and returns once the pass
// has ended, with its caller still waiting and with its caller gone. A load
// that succeeds leaves the pass encoding, since that overlap is what it is
// for, unless its caller has gone: then the pass is stopped and waited for,
// and what stopped it is no failure a later request is told of.
//
// A runtime gives the pass to the first request of a prompt that it resolves.
// That request waits for the pass, is told what the pass failed with in place
// of an encoding run a second time, and is the only one told. A request
// cancelled while it waits stops the pass and returns when it has ended, and
// so does a runtime closed with its pass still running.
func TestFailedLoadEndsItsPromptPass(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		refused := errors.New("the generator would not load")
		unread := errors.New("the encoder's checkpoint could not be read")
		request := WanRequest{Prompt: "a fox crosses the snow"}
		// held begins a pass whose encoding waits to be let go, giving found, or to be stopped; ended counts the encodings that have returned. A stopped encoding has work to finish before it returns, as an encoder between two blocks has: a caller that stopped it without waiting has come back by then.
		held := func(ctx context.Context, found error) (pass *promptPass, release chan struct{}, ended *atomic.Int32) {
			release, ended = make(chan struct{}), &atomic.Int32{}
			return beginPromptPass(ctx, func(ctx context.Context) error {
				defer ended.Add(1)
				select {
				case <-release:
					return found
				case <-ctx.Done():
					synctest.Wait()
					return ctx.Err()
				}
			}), release, ended
		}

		// A load that fails, its caller still waiting.
		pass, _, ended := held(t.Context(), nil)
		if _, err := loadedBeside(t.Context(), pass, func() (*WanRuntime, error) { return nil, refused }); err != refused || ended.Load() != 1 {
			t.Fatalf("a load that failed = %v with %d encodings ended when it came back; want what it failed with, and its pass ended", err, ended.Load())
		}

		// A load that fails, its caller gone while it loaded.
		ctx, cancel := context.WithCancel(t.Context())
		pass, _, ended = held(ctx, nil)
		if _, err := loadedBeside(ctx, pass, func() (*WanRuntime, error) { cancel(); return nil, refused }); err != refused || ended.Load() != 1 {
			t.Fatalf("a load that failed with its caller gone = %v with %d encodings ended when it came back; want what it failed with, and its pass ended", err, ended.Load())
		}

		// A load that succeeds with its caller gone: the pass is stopped and waited for, and a later request is told nothing of it.
		ctx, cancel = context.WithCancel(t.Context())
		pass, _, ended = held(ctx, nil)
		wan, err := loadedBeside(ctx, pass, func() (*WanRuntime, error) { cancel(); return &WanRuntime{pass: pass}, nil })
		if err != nil || ended.Load() != 1 {
			t.Fatalf("a load that succeeded with its caller gone = %v with %d encodings ended when it came back; want its pass ended", err, ended.Load())
		}
		if err := wan.pass.taken(t.Context()); err != nil {
			t.Fatalf("a later request was told %v by a pass its own request's going had stopped", err)
		}

		// A load that succeeds: the pass goes on encoding, and the request that resolves waits for it.
		pass, release, ended := held(t.Context(), nil)
		wan, err = loadedBeside(t.Context(), pass, func() (*WanRuntime, error) { return &WanRuntime{pass: pass}, nil })
		if err != nil || ended.Load() != 0 {
			t.Fatalf("a load that succeeded = %v with %d encodings ended when it came back; want its pass still encoding beside it", err, ended.Load())
		}
		go func() { synctest.Wait(); close(release) }()
		// This runtime is of no model, so the request's own conditioning fails, after the pass.
		if _, err := wan.resolve(t.Context(), request); err == nil || ended.Load() != 1 || wan.pass != nil {
			t.Fatalf("a request resolved beside its pass = %v with %d encodings ended; want it to have waited for the pass and taken it", err, ended.Load())
		}

		// A pass that failed: the first request is told, and only the first.
		pass, release, _ = held(t.Context(), unread)
		wan = &WanRuntime{pass: pass}
		close(release)
		if _, err := wan.resolve(t.Context(), request); !errors.Is(err, unread) {
			t.Fatalf("the request whose pass failed = %v, want what the pass failed with and no second encoding", err)
		}
		if _, err := wan.resolve(t.Context(), request); err == nil || errors.Is(err, unread) {
			t.Fatalf("the request after it = %v, want its own conditioning and not the pass's failure a second time", err)
		}

		// A request cancelled while it waits for the pass.
		pass, _, ended = held(t.Context(), nil)
		wan = &WanRuntime{pass: pass}
		ctx, cancel = context.WithCancel(t.Context())
		go func() { synctest.Wait(); cancel() }()
		if _, err := wan.resolve(ctx, request); !errors.Is(err, context.Canceled) || ended.Load() != 1 {
			t.Fatalf("a request cancelled while it waited for its pass = %v with %d encodings ended when it came back; want the cancellation, and the pass ended", err, ended.Load())
		}

		// A runtime closed with its pass still running.
		pass, _, ended = held(t.Context(), nil)
		wan = &WanRuntime{pass: pass}
		if err := wan.Close(t.Context()); err != nil || ended.Load() != 1 {
			t.Fatalf("a runtime closed with its pass running = %v with %d encodings ended when it came back; want the pass ended", err, ended.Load())
		}
	})
}
