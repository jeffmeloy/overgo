package speechsynth

import (
	"math"
	"testing"
)

// TestBackboneStreamGolden gates the flow-LM conditioner (ladder g6): one
// full causal pass over [voice; text; bos] must reproduce each reference
// streaming call's output rows (causality makes streamed and full-sequence
// computations identical). Voice conditioning enters from the golden call
// input; our own text embeddings and bos row are gated against the same
// inputs first, so the stream is fully cross-checked.
func TestBackboneStreamGolden(t *testing.T) {
	m := loadArtifactModel(t)
	g := loadFixture[g6Golden](t, "g6_backbone.json")
	if len(g.TransformerCalls) != 3 {
		t.Fatalf("want 3 transformer calls, got %d", len(g.TransformerCalls))
	}
	d := m.Dims.DModel
	tv := g.TransformerCalls[0].In.Shape[1]
	tt := g.TransformerCalls[1].In.Shape[1]
	tg := g.TransformerCalls[2].In.Shape[1]
	total := tv + tt + tg

	// Text embedding gate: g1 ids -> conditioner rows.
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")
	textEmb := make([]float32, len(g1.IDs)*d)
	if err := m.TextEmbedInto(textEmb, g1.IDs); err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "text embeddings", textEmb, g.TextEmbeddings.Values, tolTextEmbed)

	// BOS row gate: input_linear(bos_emb) must equal the third call's input.
	bosRow := make([]float32, d)
	m.LatentInputInto(bosRow, m.BosEmb)
	requireWithin(t, "bos input row", bosRow, g.TransformerCalls[2].In.Values, tolBosRow)

	// Full stream from the golden call INPUTS.
	stream := make([]float32, 0, total*d)
	stream = append(stream, f32of(g.TransformerCalls[0].In.Values)...)
	stream = append(stream, f32of(g.TransformerCalls[1].In.Values)...)
	stream = append(stream, f32of(g.TransformerCalls[2].In.Values)...)
	st := m.NewDecodeState(total)
	m.AppendForward(st, stream, total)

	segments := []struct {
		name     string
		lo, n    int
		expected []float64
	}{
		{"voice segment", 0, tv, g.TransformerCalls[0].Out.Values},
		{"text segment", tv, tt, g.TransformerCalls[1].Out.Values},
		{"gen segment", tv + tt, tg, g.TransformerCalls[2].Out.Values},
	}
	for _, s := range segments {
		requireWithin(t, s.name, stream[s.lo*d:(s.lo+s.n)*d], s.expected, tolTransformer)
	}

	// Flow conditioning row: the first flow call's cond is out_norm of the
	// VOICE segment's last row (reference-verified identity).
	normed := make([]float32, d)
	m.OutNormInto(normed, stream[(tv-1)*d:tv*d], 1)
	requireWithin(t, "flow cond (out_norm voice last row)", normed, g.FlowStep.Cond.Values, tolTransformer)

	// g3 boundary summaries corroborate the same tensors through the
	// reference's forward hooks; bounds derive from the elementwise gates.
	g3 := loadFixture[g3Golden](t, "g3_stage_traces.json")
	assertSummary(t, "g3 conditioner", g3.Traces["conditioner"], textEmb, tolTextEmbed)
	assertSummary(t, "g3 transformer", g3.Traces["transformer"], stream[:tv*d], tolTransformer)
}

// assertSummary checks rms/sum/first against a golden summary trace with
// bounds induced by the elementwise gate g: |d rms| <= g, |d sum| <= N*g,
// first slice elementwise <= g.
func assertSummary(t *testing.T, name string, want goldenSummary, got []float32, gate float64) {
	t.Helper()
	if want.Numel != len(got) {
		t.Fatalf("%s: numel %d != golden %d", name, len(got), want.Numel)
	}
	var ss, sum float64
	for _, v := range got {
		ss += float64(v) * float64(v)
		sum += float64(v)
	}
	rms := math.Sqrt(ss / float64(len(got)))
	rmsDiff := math.Abs(rms - want.RMS)
	sumDiff := math.Abs(sum - want.Sum)
	t.Logf("%s: rms diff %.6e (gate %.0e), sum diff %.6e (gate %.3g)", name, rmsDiff, gate, sumDiff, float64(want.Numel)*gate)
	if rmsDiff > gate {
		t.Fatalf("%s rms %g vs %g diverges past %g", name, rms, want.RMS, gate)
	}
	if sumDiff > float64(want.Numel)*gate {
		t.Fatalf("%s sum %g vs %g diverges past %g", name, sum, want.Sum, float64(want.Numel)*gate)
	}
	requireWithin(t, name+" first", got[:len(want.First)], want.First, gate)
}

// TestFlowOneStepGolden gates the flow head against the captured reference
// flow step (g6): F(cond, s=0, t=1, noise) matches flow_out, and the
// one-step latent is exactly noise + F.
func TestFlowOneStepGolden(t *testing.T) {
	m := loadArtifactModel(t)
	g := loadFixture[g6Golden](t, "g6_backbone.json")

	cond := f32of(g.FlowStep.Cond.Values)
	noise := f32of(g.FlowStep.NoiseIn.Values)
	out := m.FlowForward(cond, g.FlowStep.S, g.FlowStep.T, noise)
	requireWithin(t, "flow_net output", out, g.FlowStep.FlowOut.Values, tolFlowOut)

	step := make([]float32, len(noise))
	m.OneStepLatentInto(step, cond, noise)
	for i := range step {
		if want := noise[i] + out[i]; step[i] != want {
			t.Fatalf("one-step latent mismatch at %d: %g != %g", i, step[i], want)
		}
	}

	// g3 flow_net boundary summary corroborates the same call.
	g3 := loadFixture[g3Golden](t, "g3_stage_traces.json")
	assertSummary(t, "g3 flow_net", g3.Traces["flow_net"], out, tolFlowOut)
}
