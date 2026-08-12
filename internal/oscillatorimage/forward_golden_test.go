package oscillatorimage

import (
	"encoding/json"
	"testing"
)

type dynamicsGolden struct {
	B      int       `json:"b"`
	N      int       `json:"n"`
	Nc     int       `json:"nc"`
	State  []float64 `json:"state"`
	Omega  []float64 `json:"omega"`
	OmegaC []float64 `json:"omega_c"`
	K      []float64 `json:"K"`
	Kc     []float64 `json:"Kc"`
	Drive  []float64 `json:"drive"`
	Out    []float64 `json:"out"`
}

// TestDynamicsMatchesTorch: coupled conditional-Kuramoto velocity vs the
// reference torch golden (protocol: adaptive TestUn0DynamicsMatchesTorch).
func TestDynamicsMatchesTorch(t *testing.T) {
	var g dynamicsGolden
	loadReferenceGolden(t, "un0_dynamics_golden.json", &g)
	tot := g.N + g.Nc
	out := make([]float32, g.B*tot)
	trig := make([]float64, 2*tot)
	conditionalKuramotoForwardInto(out, f32of(g.State), f32of(g.Omega), f32of(g.OmegaC),
		f32of(g.K), f32of(g.Kc), f32of(g.Drive), g.B, g.N, g.Nc, 1, 1, 1, trig[:tot], trig[tot:])
	requireWithin(t, "dynamics", out, g.Out, tolOperator)
}

// TestGeneratorMatchesTorch: full dynamics -> readout -> decoder parity vs
// the reference torch golden (protocol: adaptive TestUn0GeneratorMatchesTorch).
func TestGeneratorMatchesTorch(t *testing.T) {
	var m map[string]json.RawMessage
	loadReferenceGolden(t, "un0_generator_golden.json", &m)
	gi := func(k string) int { var v int; json.Unmarshal(m[k], &v); return v }
	gf := func(k string) []float64 { var v []float64; json.Unmarshal(m[k], &v); return v }
	n, nc, b := gi("n"), gi("nc"), gi("b")
	inCh, inH, inW, outCh := gi("in_ch"), gi("in_h"), gi("in_w"), gi("out_ch")
	numSteps := gi("num_steps")
	var dt float64
	json.Unmarshal(m["dt"], &dt)

	var blocksRaw []map[string]json.RawMessage
	json.Unmarshal(m["blocks"], &blocksRaw)
	blocks := make([]DecoderBlock, len(blocksRaw))
	for i, br := range blocksRaw {
		var w1, b1, w2, b2 []float64
		var cout int
		json.Unmarshal(br["w1"], &w1)
		json.Unmarshal(br["b1"], &b1)
		json.Unmarshal(br["w2"], &w2)
		json.Unmarshal(br["b2"], &b2)
		json.Unmarshal(br["cout"], &cout)
		blocks[i] = DecoderBlock{W1: f32of(w1), B1: f32of(b1), W2: f32of(w2), B2: f32of(b2), Cout: cout}
	}

	if b != 1 {
		t.Fatalf("generator fixture batch = %d", b)
	}
	model := Model{
		Cfg: Config{
			N: n, NCond: nc, InChannels: inCh, InH: inH, InW: inW, OutChannels: outCh,
			NumSteps: numSteps, Dt: dt, KScale: 1, KCondScale: 1, KDriveScale: 1,
			Relativization: "ref_oscillator", Encoding: "sin_cos", TanhOut: true,
			BlockChannels: func() []int {
				channels := make([]int, len(blocks))
				for index := range blocks {
					channels[index] = blocks[index].Cout
				}
				return channels
			}(),
		},
		Slope: decoderNegativeSlope, Omega: f32of(gf("omega")), OmegaCond: f32of(gf("omega_cond")),
		K: f32of(gf("K")), KCond: f32of(gf("K_cond")), Blocks: blocks,
		ToOutW: f32of(gf("to_out_w")), ToOutB: f32of(gf("to_out_b")),
	}
	features, err := model.integrate(phasePlan{state: f32of(gf("init")), drive: f32of(gf("drive"))})
	if err != nil {
		t.Fatal(err)
	}
	image, err := model.decode(features)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "generator", image.Pixels, gf("out"), tolGenerator)
}
