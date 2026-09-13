package oscillatorimage

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"testing"
)

type phaseCancellation struct {
	context.Context
	checks int
	limit  int
	cancel context.CancelCauseFunc
}

func (ctx *phaseCancellation) Err() error {
	ctx.checks++
	if ctx.limit > 0 && ctx.checks >= ctx.limit {
		ctx.cancel(context.Canceled)
	}
	return ctx.Context.Err()
}

func TestStagedCancellationKeepsOutputs(t *testing.T) {
	model := tinyModel()
	request := Request{Class: 0, Seed: 42}
	plan, err := model.prepare(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	state := slices.Clone(plan.state)
	features, err := model.integrate(t.Context(), plan)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.decode(t.Context(), features)
	if err != nil {
		t.Fatal(err)
	}
	for _, phase := range []string{"prepare", "integrate", "decode"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(context.Canceled)
			var err error
			switch phase {
			case "prepare":
				_, err = model.prepare(ctx, request)
			case "integrate":
				_, err = model.integrate(ctx, plan)
			case "decode":
				_, err = model.decode(ctx, features)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled %s: %v", phase, err)
			}
			again, err := model.integrate(t.Context(), plan)
			if err != nil {
				t.Fatal(err)
			}
			got, err := model.decode(t.Context(), again)
			if err != nil || !slices.Equal(plan.state, state) || !slices.Equal(again, features) || !bytes.Equal(got.Data, want.Data) {
				t.Fatal("cancellation changed reusable input, features or PNG", err)
			}
		})
	}
	// Learn the checks exercised by one complete integration step, then stop
	// the full model at the next work boundary without a timer.
	control := *model
	control.Cfg.NumSteps = 1
	probe := &phaseCancellation{Context: t.Context()}
	if _, err := control.integrate(probe, plan); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(context.Canceled)
	interrupted := &phaseCancellation{Context: ctx, limit: probe.checks + 1, cancel: cancel}
	if _, err := model.integrate(interrupted, plan); !errors.Is(err, context.Canceled) {
		t.Fatal("integration did not stop at its work boundary", err)
	}
	again, err := model.integrate(t.Context(), plan)
	if err != nil || !slices.Equal(again, features) || !slices.Equal(plan.state, state) {
		t.Fatal("interrupted integration changed subsequent output", err)
	}
}

func TestVideoCancellationKeepsOutputs(t *testing.T) {
	model := tinyModel()
	request := VideoRequest{Class: 0, Seed: 42, Frames: model.Cfg.NClasses, Scale: 1}
	plan, err := model.prepareVideo(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	features, err := model.integrateVideo(t.Context(), plan)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.decodeVideo(t.Context(), features)
	if err != nil {
		t.Fatal(err)
	}
	for _, phase := range []string{"prepare", "integrate", "decode"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			probe := &phaseCancellation{Context: t.Context()}
			switch phase {
			case "prepare":
				cancel(context.Canceled)
			case "integrate":
				one := plan
				one.request.Frames = 1
				if _, err := model.integrateVideo(probe, one); err != nil {
					t.Fatal(err)
				}
			case "decode":
				one := features
				one.frames = features.frames[:1]
				if _, err := model.decodeVideo(probe, one); err != nil {
					t.Fatal(err)
				}
			}
			boundary := &phaseCancellation{Context: ctx, limit: probe.checks + 1, cancel: cancel}
			var err error
			switch phase {
			case "prepare":
				_, err = model.prepareVideo(boundary, request)
			case "integrate":
				_, err = model.integrateVideo(boundary, plan)
			case "decode":
				// The single-frame final encoding check is the next frame's
				// admission check in the full clip.
				boundary.limit = probe.checks
				_, err = model.decodeVideo(boundary, features)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("video %s cancellation: %v", phase, err)
			}
			again, err := model.integrateVideo(t.Context(), plan)
			if err != nil {
				t.Fatal(err)
			}
			got, err := model.decodeVideo(t.Context(), again)
			if err != nil || !bytes.Equal(got.Data, want.Data) || got.ChangedPixels != want.ChangedPixels {
				t.Fatal("video cancellation changed subsequent GIF", err)
			}
		})
	}
}
