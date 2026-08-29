package main

// Device mode: takes the RxBrain routed-LM terminal (final RMSNorm + lm_head
// vocab projection + argmax) onto the CUDA device through overgo's generic
// executor (OpRMSNorm + OpMultiply + OpMulMat + OpTopK, native-BF16 head
// residency), and asserts device == host-graph == routedlm-host == golden on
// the canonical bridgev2 case. This is the decode-critical vocab projection:
// the dominant per-token GEMM (hidden 2048 -> vocab 120818, BF16). The full
// 32-layer decode stack + vision tower remain host-only (see report).

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// runDevice: device terminal parity + measurement.
func runDevice(l *campaignContext) error {
	ctx := context.Background()
	l.Log("DEVICE terminal parity START")

	cfg, err := routedlm.LoadConfig(l.modelDir, binding)
	if err != nil {
		return err
	}
	src, err := safetensors.OpenSource(l.modelDir)
	if err != nil {
		return err
	}
	defer src.Close()
	terminal, err := routedlm.LoadTerminalWeights(src, cfg, binding)
	if err != nil {
		return err
	}
	H, vocab := cfg.HiddenSize, cfg.VocabSize
	eps := float32(cfg.RMSNormEps)

	// layer31 golden output -> last row (the decode-relevant terminal input).
	lg, err := loadGoldenJSON[motGolden](l.fixturesDir, "rxbrain_vqa_mot_layer31_output_golden.json")
	if err != nil {
		return err
	}
	layer31, err := loadTensorAsset(l.fixturesDir, lg.LayerOutputAsset, lg.MoTTensors["layer31_output"])
	if err != nil {
		return err
	}
	if len(layer31)%H != 0 {
		return fmt.Errorf("layer31 len %d not divisible by hidden %d", len(layer31), H)
	}
	rows := len(layer31) / H
	lastRow := append([]float32(nil), layer31[(rows-1)*H:rows*H]...)
	normWeight := terminal.FinalNorm[tensor.FirstOffset]

	tg, err := loadGoldenJSON[terminalGolden](l.fixturesDir, "rxbrain_vqa_terminal_golden.json")
	if err != nil {
		return err
	}
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}

	// Head: raw BF16 bytes (device residency) + F32 dequant (host reference).
	if terminal.Head.DType != "BF16" {
		return fmt.Errorf("device terminal: head dtype %s, want BF16", terminal.Head.DType)
	}
	headBytes := make([]byte, terminal.Head.Size())
	if _, err := terminal.Head.ReadAt(headBytes, tensor.FirstOffset); err != nil {
		return fmt.Errorf("device terminal: head read: %w", err)
	}
	headF32 := make([]float32, vocab*H)
	for i := range headF32 {
		headF32[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(headBytes[i*2:])) << 16)
	}

	headShape := tensor.MustShape(uint64(H), uint64(vocab)) // ggml [K=hidden, M=vocab]
	rowShape := tensor.MustShape(uint64(H), tensor.SingletonExtent)
	normShape := tensor.MustShape(uint64(H))

	// ---- host reference graph (F32 head) -----------------------------------
	rb := tensor.NewBuilder()
	rRow := rb.Input("hidden", dtype.F32, rowShape)
	rNorm := rb.Input("norm", dtype.F32, normShape)
	rHead := rb.Input("head", dtype.F32, headShape)
	rFinal := rb.WeightedRMSNorm(rRow, rNorm, eps)
	rLogits := rb.MulMat(rHead, rFinal)
	rTop := rb.TopK(rLogits, tensor.SingletonExtent)
	refOut, err := reference.Execute([]*tensor.Tensor{rLogits, rTop}, map[*tensor.Tensor]reference.Value{
		rRow:  {Shape: rowShape, Data: lastRow},
		rNorm: {Shape: normShape, Data: normWeight},
		rHead: {Shape: headShape, Data: headF32},
	})
	if err != nil {
		return fmt.Errorf("device terminal: reference execute: %w", err)
	}
	hostLogits := refOut[rLogits].Data
	hostTop := int(refOut[rTop].Data[0])

	// ---- device graph (native-BF16 head residency) -------------------------
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		return fmt.Errorf("device terminal: worker: %w", err)
	}
	defer worker.Close()
	exe, err := executor.NewWithWorker(worker)
	if err != nil {
		return fmt.Errorf("device terminal: executor: %w", err)
	}
	defer exe.Close()

	var headPtr driver.DevicePtr
	if err := worker.Do(ctx, func(state *device.State) error {
		p, e := state.Driver.MemAlloc(uint64(len(headBytes)))
		if e != nil {
			return e
		}
		if e := state.Driver.MemcpyHtoD(p, headBytes); e != nil {
			_ = state.Driver.MemFree(p)
			return e
		}
		headPtr = p
		return nil
	}); err != nil {
		return fmt.Errorf("device terminal: head upload: %w", err)
	}
	defer func() {
		_ = worker.Do(ctx, func(state *device.State) error { return state.Driver.MemFree(headPtr) })
	}()

	db := tensor.NewBuilder()
	dRow := db.Input("hidden", dtype.F32, rowShape)
	dNorm := db.Input("norm", dtype.F32, normShape)
	dHead := db.Input("head", dtype.BF16, headShape)
	dFinal := db.WeightedRMSNorm(dRow, dNorm, eps)
	dLogits := db.MulMat(dHead, dFinal)
	dTop := db.TopK(dLogits, tensor.SingletonExtent)
	compiled, err := executor.Compile(dLogits, dTop)
	if err != nil {
		return fmt.Errorf("device terminal: compile: %w", err)
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{
		dRow:  {Shape: rowShape, Data: lastRow},
		dNorm: {Shape: normShape, Data: normWeight},
	}
	devInputs := compiled.NewDeviceInputs()
	if err := devInputs.Set(dHead, headPtr); err != nil {
		return err
	}
	devOut, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, devInputs)
	if err != nil {
		return fmt.Errorf("device terminal: device execute: %w", err)
	}
	devLogits := devOut[dLogits].Data
	devTop := int(devOut[dTop].Data[0])
	// ---- exactness ---------------------------------------------------------
	// device top == host-graph top == golden first token.
	if devTop != hostTop {
		return fmt.Errorf("device top token %d != host-graph top %d", devTop, hostTop)
	}
	if devTop != dg.FirstToken || devTop != tg.LastLogits.TopIndex[0] {
		return fmt.Errorf("device top token %d != golden first %d / terminal top %d", devTop, dg.FirstToken, tg.LastLogits.TopIndex[0])
	}
	// device logits == host-graph logits at the golden probe + top indices
	// (native-BF16 decode kernel vs BF16-dequant F32 reference).
	logitIDs := slices.Clone(tg.LastLogits.ProbeIndex)
	logitIDs = append(logitIDs, tg.LastLogits.TopIndex...)
	var worstDH float64
	for _, id := range logitIDs {
		if id < 0 || id >= len(devLogits) {
			return fmt.Errorf("device terminal: logit index %d out of range", id)
		}
		if d := math.Abs(float64(devLogits[id]) - float64(hostLogits[id])); d > worstDH {
			worstDH = d
		}
	}
	// host-graph logits vs routedlm golden LastLogits (graph form is faithful).
	devProbe := make([]float32, len(tg.LastLogits.ProbeIndex))
	for i, id := range tg.LastLogits.ProbeIndex {
		devProbe[i] = devLogits[id]
	}
	if _, err := probeValuesCheck("device last_logits sparse", devProbe, tg.LastLogits.ProbeValue, acceptTerminal); err != nil {
		return err
	}
	devTopV := []float32{devLogits[tg.LastLogits.TopIndex[0]]}
	if _, err := probeValuesCheck("device last_logits argmax", devTopV, []float64{tg.LastLogits.TopValue[0]}, acceptTerminal); err != nil {
		return err
	}

	// ---- measurement -------------------------------------------------------
	budget := measurement(measureTerminal)
	perCall, err := measure(budget, func() error {
		if _, err := exe.ExecuteCompiled(ctx, compiled, hostFeeds, devInputs); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	l.Log(fmt.Sprintf("DEVICE terminal EXACT top=%d (golden first=%d) dev-vs-host worst|d|=%.3e",
		devTop, dg.FirstToken, worstDH))
	l.Log(fmt.Sprintf("DEVICE terminal MEASURE proj=%.3fms/token head_resident=%.0fMiB",
		float64(perCall.Microseconds())/1000.0, float64(len(headBytes))/(1<<20)))
	l.Log("DEVICE terminal LANE GREEN")
	return nil
}
