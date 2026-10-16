//go:build windows

package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

func TestASRResidentCUDA(t *testing.T) {
	cudatest.Require(t)
	t.Run("encoder", func(t *testing.T) {
		fixture, _, e := smallEncoder(t)
		before := driver.ProcessCurrentBytes()
		if err := e.BindDevice(t.Context(), 0); err != nil {
			t.Fatal(err)
		}
		defer e.Close(context.Background())
		static := e.resident.session.StaticBytes()
		if static == 0 {
			t.Fatal("device retained no weights")
		}
		_, _, refused := smallEncoder(t)
		for _, budget := range []uint64{static, static + 1} {
			refused.memoryBytes = budget
			allocated := driver.ProcessCurrentBytes()
			if err := refused.BindDevice(t.Context(), 0); err == nil {
				_ = refused.Close(context.Background())
				t.Fatal("accepted insufficient resident budget")
			}
			if refused.resident != nil || driver.ProcessCurrentBytes() != allocated {
				t.Fatal("failed binding retained device allocations")
			}
		}
		var w Workspace
		cases := append(slices.Clone(fixture.Cases), fixture.Cases[0])
		for _, tc := range cases {
			tolerance := float64(math.Nextafter32(1, 2)-1) * float64(e.blocks[0].first.in.out)
			hidden, frames, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, func(trace Trace) error {
				name := fmt.Sprintf("layer_%d", trace.Block)
				if trace.Block < 0 {
					name = "input_projection"
				}
				testutil.RequireWithin(t, name, trace.Values, tc.Traces[name].Values, tolerance)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			logits, err := e.Project(t.Context(), hidden, frames, &w)
			if err != nil {
				t.Fatal(err)
			}
			testutil.RequireWithin(t, "logits", logits, tc.Logits, tolerance)
			if e.resident.session.StaticBytes() != static {
				t.Fatal("changing frame count duplicated resident weights")
			}
			cancelled, cancel := context.WithCancel(t.Context())
			cancel()
			if _, _, err := e.Encode(cancelled, tc.Features, tc.Frames, &w, nil); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation: %v", err)
			}
		}
		stats, err := e.resident.session.Stats(t.Context())
		if err != nil || stats.Execution.KernelLaunches == 0 || stats.Execution.HostToDeviceBytes == 0 || stats.Execution.DeviceToHostBytes == 0 {
			t.Fatalf("CUDA execution not established: %+v %v", stats, err)
		}
		tc := fixture.Cases[0]
		cancelled, cancel := context.WithCancel(t.Context())
		defer cancel()
		if _, _, err := e.Encode(cancelled, tc.Features, tc.Frames, &w, func(trace Trace) error {
			if trace.Block >= 0 {
				cancel()
			}
			return nil
		}); !errors.Is(err, context.Canceled) {
			t.Fatalf("mid-forward cancellation: %v", err)
		}
		if err := e.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
		if driver.ProcessCurrentBytes() != before {
			t.Fatal("closing the recognizer retained device allocations")
		}
		if _, _, err := e.Encode(t.Context(), tc.Features, tc.Frames, &w, nil); err == nil {
			t.Fatal("closed binding silently executed")
		}
	})
	t.Run("stream_refused", func(t *testing.T) {
		_, _, _, model := tinyTransducer(t)
		if err := model.BindDevice(t.Context(), 0); err != nil {
			t.Fatal(err)
		}
		defer model.Close(context.Background())
		var workspace TransducerWorkspace
		if _, err := model.RecognizeChunk(t.Context(), []float32{1, -1, 2, -2}, 2, true, &workspace, nil); err == nil || workspace.stream != nil {
			t.Fatalf("resident binding admitted a native stream: %v", err)
		}
	})
	t.Run("recipe_lifetime", func(t *testing.T) { testTranscriptionRecipeLineage(t, true) })
}
