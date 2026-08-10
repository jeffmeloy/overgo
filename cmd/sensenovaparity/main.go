// sensenovaparity walks the SenseNova-U1 MoT forward path against the
// device-neutral oracle fixtures (edit_oracle_v3_256.json,
// generation_oracle_256.json) on the CPU host packages, the way cmd/vqaparity
// walks the RxBrain ladder. It is the CPU analogue of the adaptive_new
// CUDA-gated harness (sensenova_edit_cuda_windows_test.go,
// research_cuda_windows_test.go): it reads the SAME fixture JSON and asserts
// each stage the fixtures + ported host surface + the read-only checkpoint can
// honestly verify, streaming only the tensors it probes (never materializing
// the 35GB model).
//
// Honesty contract (task step 3): stages backed by a real value oracle are
// asserted (schedule knots, z geometry/sigma, plan+rope derivation). Stages
// that reach flow-forward arithmetic the fixtures do NOT cover are computed on
// the real checkpoint and reported ORACLE-ABSENT (finiteness/shape only, no
// parity claim). Stages whose oracle exists but whose INPUT is unavailable
// (sparse 64-sample probes, external source image, unported renderer/denoise)
// are reported FRONTIER with the exact blocker and the exact next stage — a
// weak check is never dressed up as parity.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

const (
	verdictOracle   = "PASS-ORACLE" // real value oracle asserted
	verdictDerive   = "PASS-DERIVE" // derivation/structural on the real checkpoint
	verdictWired    = "PASS-WIRED"  // forward computed on real weights; ORACLE-ABSENT
	verdictFrontier = "FRONTIER"    // oracle exists but input unavailable; no parity claim
	verdictFail     = "FAIL"
)

type ladder struct {
	logPath string
	failed  bool
	counts  map[string]int
}

func (l *ladder) log(line string) {
	stamp := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	full := stamp + " " + line + "\n"
	fmt.Print(full)
	if f, err := os.OpenFile(l.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
		_, _ = f.WriteString(full)
		_ = f.Close()
	}
}

// stage: runs fn unless a prior stage FAILED (frontier/oracle-absent do not
// stop the ladder — they are documented boundaries, not failures). worst is
// the worst |d| against the oracle where one exists, NaN where none does.
func (l *ladder) stage(name string, fn func() (verdict string, worst float64, detail string, err error)) {
	if l.failed {
		return
	}
	start := time.Now()
	verdict, worst, detail, err := fn()
	wall := time.Since(start).Round(time.Millisecond)
	if err != nil {
		l.failed = true
		l.counts[verdictFail]++
		l.log(fmt.Sprintf("STAGE %-26s %-11s worst|d|=%-11s wall=%-9s %v", name, verdictFail, "-", wall, err))
		return
	}
	l.counts[verdict]++
	worstStr := "n/a"
	if !math.IsNaN(worst) {
		worstStr = fmt.Sprintf("%.3e", worst)
	}
	l.log(fmt.Sprintf("STAGE %-26s %-11s worst|d|=%-11s wall=%-9s %s", name, verdict, worstStr, wall, detail))
}

func main() {
	modelDir := flag.String("model", `C:\Users\jeffm\adaptive_new\models\SenseNova-U1-8B-MoT-Infographic-V3`, "checkpoint dir (READ-ONLY)")
	fixturesDir := flag.String("fixtures", `C:\Users\jeffm\adaptive_new\fixtures\sensenova`, "fixture dir (READ-ONLY)")
	logPath := flag.String("log", `build\sensenova_parity_log.txt`, "liveness log path")
	flag.Parse()

	l := &ladder{logPath: *logPath, counts: map[string]int{}}
	// First line immediately (liveness).
	l.log(fmt.Sprintf("LADDER START sensenovaparity model=%s fixtures=%s", *modelDir, *fixturesDir))

	if err := run(l, *modelDir, *fixturesDir); err != nil {
		l.log("LADDER ERROR " + err.Error())
		os.Exit(1)
	}
	summary := fmt.Sprintf("oracle=%d derive=%d wired=%d frontier=%d",
		l.counts[verdictOracle], l.counts[verdictDerive], l.counts[verdictWired], l.counts[verdictFrontier])
	if l.failed {
		l.log("LADDER STOPPED at first FAIL (" + summary + ")")
		os.Exit(1)
	}
	l.log("LADDER COMPLETE longest-verifiable-prefix bound (" + summary + ")")
}

func run(l *ladder, modelDir, fixturesDir string) error {
	var edit editOracle
	if err := loadJSON(fixturesDir+string(os.PathSeparator)+"edit_oracle_v3_256.json", &edit); err != nil {
		return fmt.Errorf("load edit oracle: %w", err)
	}
	var gen genOracle
	if err := loadJSON(fixturesDir+string(os.PathSeparator)+"generation_oracle_256.json", &gen); err != nil {
		return fmt.Errorf("load generation oracle: %w", err)
	}

	binding := routedlm.SenseNovaBinding()
	flowBind := routedlm.SenseNovaFlowBinding()
	cfg, err := routedlm.LoadConfig(modelDir, binding)
	if err != nil {
		return fmt.Errorf("load llm config: %w", err)
	}
	flowCfg, err := routedlm.LoadFlowConfig(modelDir)
	if err != nil {
		return fmt.Errorf("load flow config: %w", err)
	}
	src, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return fmt.Errorf("open checkpoint: %w", err)
	}
	defer src.Close()

	var flowPlan routedlm.FlowPlan

	// ---- fixture integrity ------------------------------------------------
	l.stage("edit_fixture_load", func() (string, float64, string, error) {
		if edit.Schema != "adaptive_gpt.sensenova_edit_oracle/v1" {
			return "", math.NaN(), "", fmt.Errorf("schema=%q", edit.Schema)
		}
		if len(edit.Steps) != edit.Request.Steps || edit.Request.Steps != 1 {
			return "", math.NaN(), "", fmt.Errorf("steps=%d request=%d", len(edit.Steps), edit.Request.Steps)
		}
		if len(edit.SourceContract.GridHW) != 1 || len(edit.SourceContract.GridHW[0]) != 2 {
			return "", math.NaN(), "", fmt.Errorf("grid_hw=%v", edit.SourceContract.GridHW)
		}
		pixelSamples, err := edit.SourceContract.Pixels.finiteProbes()
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("source pixels: %w", err)
		}
		embedSamples, err := edit.SourceContract.Embedding.finiteProbes()
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("source embedding: %w", err)
		}
		grid := edit.SourceContract.GridHW[0]
		return verdictDerive, math.NaN(), fmt.Sprintf("grid=%dx%d tokens=%d pixel_samples=%d embed_samples=%d prefix_layers=%v",
			grid[0], grid[1], edit.SourceContract.TokenCount, pixelSamples, embedSamples, sortedLayerKeys(edit.PrefixLayers)), nil
	})

	l.stage("gen_fixture_load", func() (string, float64, string, error) {
		if gen.Schema != "adaptive_gpt.sensenova_generation_oracle/v1" {
			return "", math.NaN(), "", fmt.Errorf("schema=%q", gen.Schema)
		}
		if len(gen.Steps) != gen.Request.Steps || gen.Request.Steps != 2 {
			return "", math.NaN(), "", fmt.Errorf("steps=%d request=%d", len(gen.Steps), gen.Request.Steps)
		}
		if len(gen.TextInputs.ConditionalIDs) == 0 || len(gen.TextInputs.UnconditionalIDs) == 0 {
			return "", math.NaN(), "", fmt.Errorf("missing text ids")
		}
		zSamples, err := gen.Steps[0].Z.finiteProbes()
		if err != nil {
			return "", math.NaN(), "", fmt.Errorf("gen z probes: %w", err)
		}
		return verdictDerive, math.NaN(), fmt.Sprintf("cond_ids=%d uncond_ids=%d z_samples=%d",
			len(gen.TextInputs.ConditionalIDs), len(gen.TextInputs.UnconditionalIDs), zSamples), nil
	})

	// ---- rope plan derivation (real checkpoint) ---------------------------
	l.stage("rope_plan_derive", func() (string, float64, string, error) {
		plan, err := routedlm.CompileRopePlan(src, cfg, binding)
		if err != nil {
			return "", math.NaN(), "", err
		}
		want := []routedlm.RopeSection{
			{Width: 64, Theta: 5e6, Axis: routedlm.AxisTime},
			{Width: 32, Theta: 1e4, Axis: routedlm.AxisHeight},
			{Width: 32, Theta: 1e4, Axis: routedlm.AxisWidth},
		}
		if len(plan.Sections) != len(want) {
			return "", math.NaN(), "", fmt.Errorf("sections=%+v", plan.Sections)
		}
		total := 0
		for i, s := range plan.Sections {
			if s != want[i] {
				return "", math.NaN(), "", fmt.Errorf("section %d=%+v want %+v", i, s, want[i])
			}
			total += s.Width
		}
		if total != cfg.HeadDim {
			return "", math.NaN(), "", fmt.Errorf("sections span %d != head_dim %d", total, cfg.HeadDim)
		}
		return verdictDerive, 0, fmt.Sprintf("sections=[64/5e6/T, 32/1e4/H, 32/1e4/W] head_dim=%d (widths from k/q_norm tensors, thetas from llm_config)", cfg.HeadDim), nil
	})

	// ---- flow plan derivation (real checkpoint) ---------------------------
	l.stage("flow_plan_derive", func() (string, float64, string, error) {
		flowPlan, err = routedlm.CompileFlowPlan(src, cfg, flowCfg, flowBind)
		if err != nil {
			return "", math.NaN(), "", err
		}
		wantFlowDim := flowPlan.VisionChannels * flowPlan.VisionPatch * flowPlan.VisionPatch * flowPlan.ImageMerge * flowPlan.ImageMerge
		if flowPlan.FlowDim != wantFlowDim {
			return "", math.NaN(), "", fmt.Errorf("flow_dim=%d != channels*patch^2*merge^2=%d", flowPlan.FlowDim, wantFlowDim)
		}
		shape, err := flowPlan.ImagePlan(256, 256)
		if err != nil {
			return "", math.NaN(), "", err
		}
		if shape.Tokens != flowPlan.NoiseScaleBase {
			return "", math.NaN(), "", fmt.Errorf("256x256 tokens=%d != noise base %d", shape.Tokens, flowPlan.NoiseScaleBase)
		}
		if shape.NoiseScale != 1 || flowPlan.NormalizedNoiseScale(shape.NoiseScale) != 0.125 {
			return "", math.NaN(), "", fmt.Errorf("256x256 sigma=%g normalized=%g", shape.NoiseScale, flowPlan.NormalizedNoiseScale(shape.NoiseScale))
		}
		return verdictDerive, 0, fmt.Sprintf("hidden=%d vision_hidden=%d merge=%d flow_dim=%d freq_dim=%d tokens256=%d sigma=%g",
			flowPlan.Hidden, flowPlan.VisionHidden, flowPlan.ImageMerge, flowPlan.FlowDim, flowPlan.FrequencyDim, shape.Tokens, shape.NoiseScale), nil
	})

	// ---- schedule knots (real oracle, exact) ------------------------------
	l.stage("edit_schedule", func() (string, float64, string, error) {
		sched, err := routedlm.ShiftedFlowTimeSchedule(edit.Request.Steps, edit.Request.TimestepShift)
		if err != nil {
			return "", math.NaN(), "", err
		}
		worst := 0.0
		for i, step := range edit.Steps {
			worst = math.Max(worst, math.Abs(sched[i]-step.Timestep))
			worst = math.Max(worst, math.Abs(sched[i+1]-step.NextTimestep))
		}
		if worst > 1e-9 {
			return "", worst, "", fmt.Errorf("schedule %v vs oracle knots, worst=%.3e", sched, worst)
		}
		return verdictOracle, worst, fmt.Sprintf("shift=%g steps=%d knots=%v", edit.Request.TimestepShift, edit.Request.Steps, sched), nil
	})

	l.stage("gen_schedule", func() (string, float64, string, error) {
		sched, err := routedlm.ShiftedFlowTimeSchedule(gen.Request.Steps, gen.Request.TimestepShift)
		if err != nil {
			return "", math.NaN(), "", err
		}
		worst := 0.0
		for i, step := range gen.Steps {
			worst = math.Max(worst, math.Abs(sched[i]-step.Timestep))
			worst = math.Max(worst, math.Abs(sched[i+1]-step.NextTimestep))
		}
		if worst > 1e-9 {
			return "", worst, "", fmt.Errorf("schedule %v vs oracle knots, worst=%.3e", sched, worst)
		}
		return verdictOracle, worst, fmt.Sprintf("shift=%g steps=%d knots=%v", gen.Request.TimestepShift, gen.Request.Steps, sched), nil
	})

	// ---- z geometry + derived noise sigma (real oracle) -------------------
	l.stage("gen_z_sigma", func() (string, float64, string, error) {
		shape, err := flowPlan.ImagePlan(gen.Request.Width, gen.Request.Height)
		if err != nil {
			return "", math.NaN(), "", err
		}
		z := gen.Steps[0].Z
		if want := shape.Tokens * flowPlan.FlowDim; z.Elements != want {
			return "", math.NaN(), "", fmt.Errorf("z elements=%d != tokens*flow_dim=%d", z.Elements, want)
		}
		sigma := shape.NoiseScale
		// z = derived_sigma * randn: sample std certifies the derived sigma.
		worst := math.Abs(z.Std - sigma)
		if worst > 0.01*sigma {
			return "", worst, "", fmt.Errorf("z std=%g vs derived sigma=%g (worst=%.3e > %.3e)", z.Std, sigma, worst, 0.01*sigma)
		}
		return verdictOracle, worst, fmt.Sprintf("z elements=%d std=%.5f derived_sigma=%g (256x256 tokens=%d)", z.Elements, z.Std, sigma, shape.Tokens), nil
	})

	l.stage("edit_z_geometry", func() (string, float64, string, error) {
		shape, err := flowPlan.ImagePlan(edit.Request.Width, edit.Request.Height)
		if err != nil {
			return "", math.NaN(), "", err
		}
		zElems := shape.Tokens * flowPlan.FlowDim // planar-patch flow state
		bElems := shape.Tokens * flowPlan.Hidden  // generation-branch hidden boundary
		step := edit.Steps[0]
		for _, c := range []struct {
			name string
			got  int
			want int
		}{
			{"z", step.Z.Elements, zElems},
			{"guided_velocity", step.GuidedVelocity.Elements, zElems},
			{"next_z", step.NextZ.Elements, zElems},
			{"conditional_boundary", step.ConditionalBoundary.Elements, bElems},
			{"source_only_boundary", step.SourceOnlyBoundary.Elements, bElems},
		} {
			if c.got != c.want {
				return "", math.NaN(), "", fmt.Errorf("%s elements=%d != derived %d", c.name, c.got, c.want)
			}
		}
		return verdictDerive, 0, fmt.Sprintf("z/vel/next_z=%d (tokens*flow_dim) boundaries=%d (tokens*hidden) tokens=%d", zElems, bElems, shape.Tokens), nil
	})

	// ---- source-contract geometry (real oracle, structural) ---------------
	l.stage("edit_source_geometry", func() (string, float64, string, error) {
		grid := edit.SourceContract.GridHW[0]
		if grid[0] != grid[1] {
			return "", math.NaN(), "", fmt.Errorf("non-square source grid %v", grid)
		}
		wantTokens := (grid[0] / flowPlan.ImageMerge) * (grid[1] / flowPlan.ImageMerge)
		if wantTokens != edit.SourceContract.TokenCount {
			return "", math.NaN(), "", fmt.Errorf("token_count=%d != (grid/merge)^2=%d", edit.SourceContract.TokenCount, wantTokens)
		}
		// pixels row width = channels*patch^2 (the patch-gather vector length).
		if len(edit.SourceContract.Pixels.Shape) == 2 {
			wantRow := flowPlan.VisionChannels * flowPlan.VisionPatch * flowPlan.VisionPatch
			if edit.SourceContract.Pixels.Shape[1] != wantRow {
				return "", math.NaN(), "", fmt.Errorf("pixels row=%d != channels*patch^2=%d", edit.SourceContract.Pixels.Shape[1], wantRow)
			}
		}
		sourceImage := grid[0] * flowPlan.VisionPatch
		return verdictDerive, 0, fmt.Sprintf("grid=%dx%d merge=%d -> tokens=%d (source image %dx%d px)",
			grid[0], grid[1], flowPlan.ImageMerge, wantTokens, sourceImage, sourceImage), nil
	})

	// ---- flow terminal wiring (ORACLE-ABSENT: no fixture for these) --------
	l.stage("flow_terminal_wiring", func() (string, float64, string, error) {
		weights, err := routedlm.LoadFlowTerminalWeights(src, flowPlan, flowBind)
		if err != nil {
			return "", math.NaN(), "", err
		}
		shape, err := flowPlan.ImagePlan(256, 256)
		if err != nil {
			return "", math.NaN(), "", err
		}
		cond, err := routedlm.FlowConditionRow(weights, flowPlan, 0, flowPlan.NormalizedNoiseScale(shape.NoiseScale))
		if err != nil {
			return "", math.NaN(), "", err
		}
		if len(cond) != flowPlan.Hidden {
			return "", math.NaN(), "", fmt.Errorf("condition row len=%d != hidden %d", len(cond), flowPlan.Hidden)
		}
		nonzero := 0
		for _, v := range cond {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return "", math.NaN(), "", fmt.Errorf("condition row has non-finite value")
			}
			if v != 0 {
				nonzero++
			}
		}
		if nonzero == 0 {
			return "", math.NaN(), "", fmt.Errorf("condition row all zeros")
		}
		// flow head on one real-width row (deterministic bf16 input, zero z).
		hidden := make([]float32, flowPlan.Hidden)
		for i := range hidden {
			hidden[i] = dtype.RoundBF16(float32(i%13)/13 - 0.5)
		}
		velocity, err := routedlm.FlowHeadVelocity(weights.Head, flowPlan, hidden, make([]float32, flowPlan.FlowDim), 0)
		if err != nil {
			return "", math.NaN(), "", err
		}
		if len(velocity) != flowPlan.FlowDim {
			return "", math.NaN(), "", fmt.Errorf("velocity len=%d != flow_dim %d", len(velocity), flowPlan.FlowDim)
		}
		for _, v := range velocity {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return "", math.NaN(), "", fmt.Errorf("velocity has non-finite value")
			}
		}
		return verdictWired, math.NaN(), fmt.Sprintf("ORACLE-ABSENT: condition row (%d nonzero/%d) + flow head (len=%d) forward finite on real weights; fixtures carry no condition/head-output tensor", nonzero, len(cond), len(velocity)), nil
	})

	// ---- vision embedder wiring (ORACLE-ABSENT for value; input sampled) ---
	l.stage("vision_embedder_wiring", func() (string, float64, string, error) {
		for _, prefix := range []struct {
			name string
			p    string
		}{
			{"source(understanding)", flowBind.SourceVisionPrefix},
			{"generation(mot_gen)", flowBind.GenerationVisionPrefix},
		} {
			if _, err := routedlm.LoadVisionEmbedderWeights(src, prefix.p, flowPlan); err != nil {
				return "", math.NaN(), "", fmt.Errorf("%s embedder: %w", prefix.name, err)
			}
		}
		return verdictWired, math.NaN(), fmt.Sprintf("ORACLE-ABSENT: both conv embedders load+shape-validate on real checkpoint; value parity BLOCKED — source_contract.pixels is a %d/%d-sample probe (need external source image to reconstruct the embedder input)",
			edit.SourceContract.Pixels.sampleCount(), edit.SourceContract.Pixels.Elements), nil
	})

	// ---- FRONTIER: text_inputs (real sha256 oracle; renderer not ported) --
	l.stage("text_inputs_frontier", func() (string, float64, string, error) {
		c := edit.TextInputs.Conditional
		s := edit.TextInputs.SourceOnly
		detail := fmt.Sprintf("FRONTIER next-stage: reconstruct+sha256 ids/time/height/width (conditional=%d, source_only=%d tokens). "+
			"BLOCKED: SenseNova edit prompt renderer (compileCausalPrefixInput) not ported — needs (a) repodb prefix-template facts (system/user/assistant/separator/image tokens) absent from overgo repodb-store, (b) a Qwen2 BPE encoder over the chat specials. "+
			"Positions (time/height/width) are already ported (routedlm.BlockPositions); only the id sequence + template facts are missing. sha256[cond.ids]=%.12s...",
			c.IDs.Elements, s.IDs.Elements, c.IDs.SHA256LEI64)
		return verdictFrontier, math.NaN(), detail, nil
	})

	// ---- FRONTIER: prefix / KV / generation forward parity ----------------
	l.stage("prefix_forward_frontier", func() (string, float64, string, error) {
		pk := edit.PrefixKV["0"].Conditional.Keys
		pl := edit.PrefixLayers["0"].Conditional
		detail := fmt.Sprintf("FRONTIER: prefix_layers/prefix_kv/generation_layers parity (probes at layers %v). "+
			"BLOCKED (three prerequisites): (1) prefix input = text embeds + source vision embeds spliced, but source embedding is a %d/%d-sample probe (full input needs the external source PNG); "+
			"(2) block-causal + multi-axis + branch-routed prefix layer-forward NOT ported (routedlm.PromptLayerForward is single-axis Time + VisualSegments windows; the pieces CompileRopePlan/BlockPositions/BlockCausalWindows exist but are not yet wired into a layer stack); "+
			"(3) generation-branch denoise stack (cross-attn to prefix KV + condition inject + flow head) NOT ported. Each probe is sparse (kv=%d/%d, hidden=%d/%d), so a FULL 42-layer/%d-token forward is required to evaluate any single probe.",
			sortedLayerKeys(edit.PrefixLayers), edit.SourceContract.Embedding.sampleCount(), edit.SourceContract.Embedding.Elements,
			pk.sampleCount(), pk.Elements, pl.sampleCount(), pl.Elements, edit.TextInputs.Conditional.IDs.Elements)
		return verdictFrontier, math.NaN(), detail, nil
	})

	// ---- FRONTIER: guided velocity / next_z (generation oracle) ------------
	l.stage("velocity_frontier", func() (string, float64, string, error) {
		detail := "FRONTIER: guided velocity + next_z parity. BLOCKED: needs the full generation pipeline output (see prefix_forward_frontier) AND a torch-matched randn(seed) for z (TorchCUDARandn not ported). The generation oracle carries only terminal probes (no intermediate layer probes), so a failure could not be localized even if the pipeline were ported."
		return verdictFrontier, math.NaN(), detail, nil
	})

	return nil
}

func sortedLayerKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// small maps (3 entries: 0,1,41); simple insertion sort by numeric value.
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && atoiSafe(keys[j-1]) > atoiSafe(keys[j]); j-- {
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	return keys
}

func atoiSafe(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 1 << 30
		}
		n = n*10 + int(c-'0')
	}
	return n
}
