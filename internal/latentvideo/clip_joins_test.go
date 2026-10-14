package latentvideo

import (
	"bytes"
	"context"
	"errors"
	"image"
	"runtime"
	"sync/atomic"
	"testing"

	"overgo/internal/media"
)

// TestVideoWorkIsJoinedOnEveryExit holds the workers that draw a clip's
// frames to the generation that hands the frames over: however the
// generation ends, it has ended only when they have.
//
// A generation that fails after a frame was taken returns its own failure
// and no clip, and returns once the frame it handed over is drawn: the
// worker, held in the middle of its drawing, has finished when the clip's
// call comes back. So has one whose request is cancelled.
//
// The frames being drawn at once are bounded, each holding the bytes taken of
// it. With every worker busy the next frame waits to be taken; a request
// cancelled while it waits has that frame refused with the cancellation, the
// frame not taken, and the clip of the frames that were taken is still the
// clip those frames make, byte for byte, drawn one after another.
func TestVideoWorkIsJoinedOnEveryExit(t *testing.T) {
	const height, width, fps, workers = 8, 12, 16, 2
	frame := func(shade float32) []float32 {
		pixels := make([]float32, media.RGBChannels*height*width)
		for at := range pixels {
			pixels[at] = shade * float32(at%width) / width
		}
		return pixels
	}
	// held is an encoder whose workers wait at gate in the middle of a drawing; entered counts those that have begun and drawn those that have ended.
	held := func() (encoder *gifEncoder, gate chan struct{}, entered, drawn *atomic.Int32) {
		t.Helper()
		encoder, err := NewClipEncoder(fps, UnitPixels)
		if err != nil {
			t.Fatal(err)
		}
		gate, entered, drawn = make(chan struct{}), &atomic.Int32{}, &atomic.Int32{}
		encoder.slots = make(chan struct{}, workers)
		encoder.draw = func(raw []uint8, height, width int, cut bool) *image.Paletted {
			entered.Add(1)
			<-gate
			defer drawn.Add(1)
			return drawFrame(raw, height, width, cut)
		}
		return encoder, gate, entered, drawn
	}
	// released opens the gate once so many workers have begun, and then only after the scheduler has run everything else that could run: a call that did not wait for its workers has come back by then.
	released := func(gate chan struct{}, entered *atomic.Int32, begun int32) {
		go func() {
			for entered.Load() < begun {
				runtime.Gosched()
			}
			for range 1 << 12 {
				runtime.Gosched()
			}
			close(gate)
		}()
	}

	refused := errors.New("the decoder failed")
	for name, ending := range map[string]func(ctx context.Context, stop context.CancelCauseFunc) error{
		"a generation that fails": func(context.Context, context.CancelCauseFunc) error { return refused },
		"a request that is cancelled": func(ctx context.Context, stop context.CancelCauseFunc) error {
			stop(refused)
			return context.Cause(ctx)
		},
	} {
		encoder, gate, entered, drawn := held()
		ctx, stop := context.WithCancelCause(t.Context())
		released(gate, entered, 1)
		video, err := writeClip(ctx, encoder, func(sink VideoFrameSink) error {
			if err := sink(0, frame(1), height, width); err != nil {
				return err
			}
			return ending(ctx, stop)
		})
		stop(nil)
		if !errors.Is(err, refused) || video.Data != nil || drawn.Load() != 1 {
			t.Fatalf("%s after a frame was taken = %v with a clip of %d bytes and %d frames drawn when it came back; want its own failure, no clip, and the frame drawn", name, err, len(video.Data), drawn.Load())
		}
	}

	// Every worker busy: the next frame waits, and a cancelled request's frame is not taken.
	encoder, gate, entered, _ := held()
	ctx, stop := context.WithCancelCause(t.Context())
	for index := range workers {
		if err := encoder.add(ctx, index, frame(float32(index+1)/workers), height, width); err != nil {
			t.Fatal(err)
		}
	}
	for entered.Load() < workers {
		runtime.Gosched()
	}
	waited := make(chan error, 1)
	go func() { waited <- encoder.add(ctx, workers, frame(0), height, width) }()
	// The scheduler has run whatever could run: a frame that did not wait has a worker drawing it by now.
	for range 1 << 12 {
		runtime.Gosched()
	}
	if begun := entered.Load(); begun != workers {
		t.Fatalf("with every worker busy a further frame was taken, %d drawings begun; want it to wait for one of the %d", begun, workers)
	}
	stop(refused)
	if err := <-waited; !errors.Is(err, refused) || len(encoder.frames) != workers {
		t.Fatalf("a frame waiting for a worker when its request is cancelled = %v with %d frames taken, want it refused with the cancellation and the %d before it taken", err, len(encoder.frames), workers)
	}
	close(gate)
	bounded, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	inTurn, err := NewClipEncoder(fps, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	for index := range workers {
		if err := inTurn.Add(index, frame(float32(index+1)/workers), height, width); err != nil {
			t.Fatal(err)
		}
	}
	want, err := inTurn.Finish()
	if err != nil || !bytes.Equal(bounded.Data, want.Data) || bounded.Frames != workers {
		t.Fatalf("the clip of the frames taken is %d bytes of %d frames, and those frames make %d bytes (%v)", len(bounded.Data), bounded.Frames, len(want.Data), err)
	}
}
