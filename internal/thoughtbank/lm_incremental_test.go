package thoughtbank

import (
	"math/rand"
	"testing"
)

// Token-for-token oracle for the incremental decode: FastWeightBankLMDecodeInit
// + FastWeightBankLMDecodeStep must reproduce FastWeightBankLMForward's
// per-position logits AND carried bank on a small deterministic fixture. The
// incremental path replays the exact per-token arithmetic in the reference's
// order, so the bound is tight (well under 1e-4, in practice bit-exact). The
// real-weights checkpoint parity test stays external-prereq (needs the shipped
// 386M weights + the SmolLM2 tokenizer).

type fixtureDecodeConfig struct {
	vocab, dModel, nHC                     int
	nHeads, dHead, nIdxHeads, nGroups      int
	csaM, hcaM, nWin, dLatentQ, topKCSA    int
	nExperts, nShared, topKExperts, dFF    int
	memDim, maxMem, memReadRank, sinkIters int
}

func smallDecodeConfig() fixtureDecodeConfig {
	return fixtureDecodeConfig{
		vocab: 16, dModel: 8, nHC: 2,
		nHeads: 4, dHead: 4, nIdxHeads: 2, nGroups: 2,
		csaM: 2, hcaM: 3, nWin: 3, dLatentQ: 6, topKCSA: 2,
		nExperts: 4, nShared: 1, topKExperts: 2, dFF: 6,
		memDim: 5, maxMem: 4, memReadRank: 2, sinkIters: 20,
	}
}

func buildFixtureAttn(rng *rand.Rand, c fixtureDecodeConfig, sparse bool) *CompressedHybridAttnWeights {
	d, dh, nh := c.dModel, c.dHead, c.nHeads
	m := c.hcaM
	if sparse {
		m = c.csaM
	}
	a := &CompressedHybridAttnWeights{
		Sparse: sparse,
		DModel: d, NHeads: nh, DHead: dh,
		M: m, NWin: c.nWin, DLatentQ: c.dLatentQ, NGroups: c.nGroups,
		TopK: c.topKCSA, NIdxHeads: c.nIdxHeads,
		WDq:        randVec(rng, c.dLatentQ*d, 0.2),
		WUq:        randVec(rng, nh*dh*c.dLatentQ, 0.2),
		WWk:        randVec(rng, dh*d, 0.2),
		WWv:        randVec(rng, dh*d, 0.2),
		OutProj:    randVec(rng, d*d, 0.2),
		QNorm:      randVec(rng, dh, 0.5),
		KVNorm:     randVec(rng, dh, 0.5),
		SinkLogits: randVec(rng, nh, 0.5),
		NormEps:    referenceRMSNormEps,
	}
	for i := 0; i < dh; i++ {
		a.QNorm[i] += 1
		a.KVNorm[i] += 1
	}
	hpg := nh / c.nGroups
	dg := d / c.nGroups
	for g := 0; g < c.nGroups; g++ {
		a.OutGroup = append(a.OutGroup, randVec(rng, dg*hpg*dh, 0.2))
	}
	if sparse {
		a.WKVa = randVec(rng, dh*d, 0.2)
		a.WKVb = randVec(rng, dh*d, 0.2)
		a.WZa = randVec(rng, dh*d, 0.2)
		a.WZb = randVec(rng, dh*d, 0.2)
		a.PosA = randVec(rng, m*dh, 0.2)
		a.PosB = randVec(rng, m*dh, 0.2)
		a.WIq = randVec(rng, c.nIdxHeads*dh*c.dLatentQ, 0.2)
		a.WW = randVec(rng, c.nIdxHeads*d, 0.2)
	} else {
		a.WKV = randVec(rng, dh*d, 0.2)
		a.WZ = randVec(rng, dh*d, 0.2)
		a.Pos = randVec(rng, m*dh, 0.2)
	}
	return a
}

func buildFixtureMHC(rng *rand.Rand, c fixtureDecodeConfig) *HyperConnectionWeights {
	n, d := c.nHC, c.dModel
	flat := n * d
	return &HyperConnectionWeights{
		NHC: n, DModel: d, SinkhornIters: c.sinkIters,
		WPre:     randVec(rng, n*flat, 0.2),
		WRes:     randVec(rng, n*n*flat, 0.2),
		WPost:    randVec(rng, n*flat, 0.2),
		SPre:     randVec(rng, n, 0.1),
		SRes:     randVec(rng, n*n, 0.1),
		SPost:    randVec(rng, n, 0.1),
		AlphaPre: 0.3, AlphaRes: 0.3, AlphaPost: 0.3,
		NormWeight: randVec(rng, flat, 0.5),
		NormEps:    referenceRMSNormEps,
	}
}

func buildFixtureModel(seed int64) (*FastWeightBankLMWeights, fixtureDecodeConfig) {
	c := smallDecodeConfig()
	rng := rand.New(rand.NewSource(seed))
	d, md := c.dModel, c.memDim
	m := &FastWeightBankLMWeights{
		VocabSize: c.vocab, DModel: d, NLayers: 2,
		NHC: c.nHC, MemDim: md, MaxMem: c.maxMem,
		Embed:   randVec(rng, c.vocab*d, 0.3),
		AOutNet: randVec(rng, c.nHC*d, 0.2),
		NormOut: randVec(rng, d, 0.5),
		Write: &thoughtWriteWeights{
			WriteCtxQ:     randVec(rng, d, 0.2),
			WriteGate:     randVec(rng, md*d, 0.2),
			WriteGateBias: randVec(rng, md, 0.1),
			ThoughtHead:   randVec(rng, md*d, 0.2),
			NormWrite:     randVec(rng, md, 0.5),
			WriteDecision: randVec(rng, d, 0.2),
			WriteDecBias:  randVec(rng, 1, 0.1),
		},
		NormEps: referenceRMSNormEps,
	}
	for i := 0; i < d; i++ {
		m.NormOut[i] += 1
	}
	for l := 0; l < 2; l++ {
		sparse := l%2 == 0
		moe := &sharedRoutedMoEWeights{
			DModel: d, DFF: c.dFF, NExperts: c.nExperts,
			NShared: c.nShared, TopK: c.topKExperts,
			WGate: randVec(rng, c.nExperts*d, 0.2),
		}
		for e := 0; e < c.nExperts; e++ {
			moe.ExpertW12 = append(moe.ExpertW12, randVec(rng, 2*c.dFF*d, 0.2))
			moe.ExpertW3 = append(moe.ExpertW3, randVec(rng, d*c.dFF, 0.2))
		}
		for s := 0; s < c.nShared; s++ {
			moe.SharedW12 = append(moe.SharedW12, randVec(rng, 2*c.dFF*d, 0.2))
			moe.SharedW3 = append(moe.SharedW3, randVec(rng, d*c.dFF, 0.2))
		}
		blk := &HyperConnectionBlockWeights{
			NHC: c.nHC, DModel: d, Attn: buildFixtureAttn(rng, c, sparse), MoE: moe,
			Bank: &FastWeightBankWeights{
				DModel: d, MemDim: md, Rank: c.memReadRank, SwiGLU: true,
				FWA:        randVec(rng, 2*c.memReadRank*d*md, 0.2),
				FWB:        randVec(rng, d*c.memReadRank*md, 0.2),
				FWO:        randVec(rng, d*d, 0.2),
				NormWeight: randVec(rng, d, 0.5),
				NormEps:    referenceRMSNormEps,
			},
			MHCAttn:   buildFixtureMHC(rng, c),
			MHCMoE:    buildFixtureMHC(rng, c),
			NormAttn:  randVec(rng, d, 0.5),
			NormMoE:   randVec(rng, d, 0.5),
			ACrossNet: randVec(rng, c.nHC*d, 0.2),
			ReadBank:  true,
			NormEps:   referenceRMSNormEps,
		}
		for i := 0; i < d; i++ {
			blk.NormAttn[i] += 1
			blk.NormMoE[i] += 1
			blk.Bank.NormWeight[i] += 1
		}
		m.Blocks = append(m.Blocks, blk)
	}
	return m, c
}

// runDecodeOracle runs the reference full forward and the incremental decode over
// the same ids/bank and asserts per-position logit parity and carried-bank
// parity.
func runDecodeOracle(t *testing.T, tag string, w *FastWeightBankLMWeights, ids []int32, bank []float32, slots int) {
	t.Helper()
	ref, err := FastWeightBankLMForward(ids, bank, slots, w)
	if err != nil {
		t.Fatalf("%s: reference forward: %v", tag, err)
	}
	vocab := w.VocabSize

	prompt := 1
	if len(ids) > 3 {
		prompt = len(ids) / 2
	}
	_, lastPrompt, err := FastWeightBankLMDecodeInit(w, ids[:prompt], bank, slots)
	if err != nil {
		t.Fatalf("%s: decode init: %v", tag, err)
	}

	got := make([][]float32, len(ids))
	state2, first, err := FastWeightBankLMDecodeInit(w, ids[:1], bank, slots)
	if err != nil {
		t.Fatalf("%s: decode init(1): %v", tag, err)
	}
	got[0] = first
	for p := 1; p < len(ids); p++ {
		l, err := FastWeightBankLMDecodeStep(state2, ids[p])
		if err != nil {
			t.Fatalf("%s: decode step %d: %v", tag, p, err)
		}
		got[p] = l
	}
	if m := maxAbsDiff(lastPrompt, got[prompt-1]); m > 1e-6 {
		t.Fatalf("%s: init(prompt) and stepped logits disagree at pos %d by %.3g", tag, prompt-1, m)
	}

	var worst float64
	argMismatch := 0
	for tpos := 0; tpos < len(ids); tpos++ {
		want := ref.Logits[tpos*vocab : (tpos+1)*vocab]
		if m := maxAbsDiff(got[tpos], want); m > worst {
			worst = m
		}
		if argmaxF32(got[tpos]) != argmaxF32(want) {
			argMismatch++
			if argMismatch <= 3 {
				t.Errorf("%s: pos %d greedy token %d, reference %d", tag, tpos, argmaxF32(got[tpos]), argmaxF32(want))
			}
		}
	}
	if argMismatch > 0 {
		t.Errorf("%s: %d/%d positions disagree on greedy next-token", tag, argMismatch, len(ids))
	}
	if worst > 1e-4 {
		t.Errorf("%s: per-position logits max abs diff %.3g exceeds 1e-4", tag, worst)
	}

	gotBank, gotSlots, err := state2.MemBank()
	if err != nil {
		t.Fatalf("%s: decode bank: %v", tag, err)
	}
	if gotSlots != ref.Slots {
		t.Fatalf("%s: decode bank %d slots, reference %d", tag, gotSlots, ref.Slots)
	}
	bankDiff := maxAbsDiff(gotBank, ref.MemBank)
	if bankDiff > 1e-4 {
		t.Errorf("%s: carried bank max abs diff %.3g exceeds 1e-4", tag, bankDiff)
	}
	if !t.Failed() {
		t.Logf("%s: %d tokens, vocab %d -- per-position logits max abs diff %.3g, bank %.3g, greedy exact %d/%d",
			tag, len(ids), vocab, worst, bankDiff, len(ids), len(ids))
	}
}

// TestFastWeightBankLMIncrementalDecodeFixture is the primary oracle: a small
// deterministic model, a sequence long enough to fill blocks and the window in
// both attention variants, with a non-empty seed bank so the per-layer read is
// live at every position.
func TestFastWeightBankLMIncrementalDecodeFixture(t *testing.T) {
	w, c := buildFixtureModel(1)
	rng := rand.New(rand.NewSource(99))

	seq := 9
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32(rng.Intn(c.vocab))
	}
	slots := 3
	bank := randVec(rng, slots*c.memDim, 0.3)

	runDecodeOracle(t, "fixture", w, ids, bank, slots)

	// Also with an EMPTY bank, so the read is skipped -- a different code path.
	runDecodeOracle(t, "fixture-nobank", w, ids, nil, 0)
}

// TestFastWeightBankLMIncrementalDecodeLengths sweeps sequence lengths across the
// first several block boundaries of both layers (CSA m=2, HCA m=3) and the
// window width (n_win=3), catching off-by-one errors in block completion or the
// ring that a single length would miss.
func TestFastWeightBankLMIncrementalDecodeLengths(t *testing.T) {
	w, c := buildFixtureModel(7)
	slots := 2
	rng := rand.New(rand.NewSource(3))
	bank := randVec(rng, slots*c.memDim, 0.3)
	for seq := 1; seq <= 12; seq++ {
		ids := make([]int32, seq)
		for i := range ids {
			ids[i] = int32((i*5 + 1) % c.vocab)
		}
		runDecodeOracle(t, "len", w, ids, bank, slots)
	}
}
