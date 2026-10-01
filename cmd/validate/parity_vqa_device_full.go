package main

// Device full-pipeline mode: chains the RxBrain stages entirely on the CUDA
// device with NO golden-seeded state — device vision tower -> device merger
// (image features) -> host embedding splice -> device branch-routed prefill (32
// chained MoT layers, exporting the per-layer resident K/V) -> terminal (first
// token) -> device 32-layer decode seeded from the DEVICE prefill K/V -> N-step
// generation. Verified to reproduce the exact answer and the golden 12-step
// chain, then measured end-to-end (wall + peak MiB) against adaptive's
// per-request bar (e2e 18.4-23.1s, engine peak 11.97GB). The lever is
// persistent residency: decode weights + the prefilled KV stay resident, so the
// decode steps replay one compiled graph.
//
// vqaserve.RunPipeline is the single owner of the stage orchestration. This
// harness drives it with golden pixel_values and the golden chain as the
// per-step verifier; the activated recipe serve (runRecipeServe) and the
// media capability drive it with processor-derived pixel_values and a
// decode-until-EOS budget.

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/vqaserve"
)

func runDeviceFull(l *campaignContext) error {
	ctx := context.Background()
	pc, err := loadPrefillContext(l)
	if err != nil {
		return err
	}
	defer pc.Src.Close()

	// golden pixel_values (harness input) + per-step golden verifier.
	pgv, err := loadGoldenJSON[processorGolden](l.fixturesDir, "rxbrain_vqa_processor_golden.json")
	if err != nil {
		return err
	}
	pixelValues, err := loadTensorAsset(l.fixturesDir, pgv.PixelValuesAsset, pgv.PixelValues)
	if err != nil {
		return err
	}
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}

	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return fmt.Errorf("device full worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device full executor: %w", err)
	}
	defer exe.Close()

	res, err := vqaserve.RunPipeline(ctx, worker, exe, pc, pixelValues, vqaserve.Options{Expected: dg.GeneratedTokens, Log: l.Log})
	if err != nil {
		return err
	}
	if err := requireIntSliceEqual("device full chain", res.Generated, dg.GeneratedTokens); err != nil {
		return err
	}
	text, err := vqaserve.DecodeChain(l.modelDir, res.Generated)
	if err != nil {
		return err
	}
	const want = "The stovetop holds a metal pot on the left burner"
	if text != want {
		return fmt.Errorf("device full decoded text %q != %q", text, want)
	}
	l.Log(fmt.Sprintf("DEVICE full ANSWER EXACT chain=%v", res.Generated))
	l.Log(fmt.Sprintf("DEVICE full TEXT %q", text))
	l.Log(fmt.Sprintf("DEVICE full MEASURE e2e=%s (vision+merger+prefill+decode) decode=%s (%d steps, %.3f ms/token)",
		res.E2EWall.Round(time.Millisecond), res.DecodeWall.Round(time.Millisecond), res.Steps, float64(res.DecodeWall.Microseconds())/1000.0/float64(res.Steps)))
	l.Log(fmt.Sprintf("DEVICE full REPLAY decode graph_launches=%d graph_instantiations=%d graph_updates=%d over %d steps (single compiled decode graph, per-step runtime attrs only)",
		res.Launches, res.Instantiations, res.Updates, res.Steps))
	l.Log(fmt.Sprintf("DEVICE full vs ADAPTIVE e2e bar 18.4-23.1s: overgo=%s (golden-seeded KV REPLACED by device prefill)", res.E2EWall.Round(time.Millisecond)))
	l.Log("DEVICE full LANE GREEN")
	return nil
}
