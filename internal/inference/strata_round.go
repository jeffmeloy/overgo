package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/hostexpert"
	"overgo/internal/model"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

// Strata's decode round for Flash-Next (strata-port/decode): the verifier (executor.StrataRound) over the sequence's
// state where it sits, its host pool (strataRoundPool), and generate.cpp's round around them. The round reads the
// weights as Strata's pack holds them: the GGUF's quantized and bf16 tensors as stored, the router, the shared expert's
// gate input, ssm_alpha, ssm_beta and the hyper-connection injects as bf16 (every value bf16-exact in Flash-Next's
// file), the PLE value projection as bf16 and its conv as f16 (neither exact: they wait for Flash-Next's held-out
// check); the round's stores, CPU pool and verifier are made at its first decode.

// strataRound is a runner's round: its bf16 and f16 stores, its CPU pool, and its verifier with the spec it holds.
type strataRound struct {
	bf16, f16 *modeldevice.DeviceConvertedWeights
	cpu       *hostexpert.Pool
	round     *executor.StrataRound
	// drafter is the round's draft layer over the runner's pack, drafterCells the cells its K/V holds.
	drafter      *executor.StrataDrafter
	drafterCells int64
	// The adaptive tier and PcieAuto (strata_round_tier.go): every expert's routes over the decode, whether swaps'
	// copies are in flight, the decode windows so far, the PCIe share as it stands and its refinement.
	usage    []float32
	swapping bool
	windows  int64
	share    float64
	pcie     strataPCIeAuto
	// promptLast holds the last prompt row read's streams (at promptLastBase), whose cell the first draft catches up.
	promptLast     *executor.DeviceBuffer
	promptLastBase driver.DevicePtr
	spec           executor.StrataRoundSpec
	pool           *strataRoundPool
	// previous is the n-gram token history before the next window (ss.ple_prev).
	previous []uint32
}

// The round's geometry, Flash-Next's, which its kernels are compiled for (verify.cpp init's checks).
const (
	strataRoundEmbedding = 2560
	strataRoundStreams   = 4
	strataRoundRank      = 320
	strataRoundExperts   = 512
	strataRoundRouted    = 10
	strataRoundExpertFF  = 640
	strataRoundStateSize = 128
	strataRoundConvTaps  = 4
	// strataRoundMaxTokens is the verify window's most tokens, S: --spec 4 widened by the suffix drafter to spec + 2;
	// strataRoundMTPTokens the MTP's windows, S_mtp: --spec.
	strataRoundMaxTokens  = 6
	strataRoundMTPTokens  = 4
	strataRoundEmbedBytes = strataRoundEmbedding / strataRoundQ80Values * strataRoundQ80Bytes
	strataRoundQ80Values  = 32 // a Q8_0 block's values, and its bytes
	strataRoundQ80Bytes   = 34
)

// strataRoundModel reports whether the runner decodes through Strata's round: a QSA runner stepping its windows by
// Strata's verifier, its experts in the file-backed residency, at Flash-Next's geometry.
func (r *Runner) strataRoundModel() bool {
	return r.qsaWindows && r.rawWeights.HasExpertResidency() && strataRoundSpec(r.spec)
}

// strataRoundSpec reports whether a model's facts are Flash-Next's, the geometry the round's kernels are compiled for.
func strataRoundSpec(s model.Spec) bool {
	return s.NGram != nil && s.EmbeddingLength == strataRoundEmbedding && s.HyperConnectionCount == strataRoundStreams &&
		s.HyperConnectionRank == strataRoundRank && s.ExpertCount == strataRoundExperts &&
		s.ExpertUsedCount == strataRoundRouted && s.ExpertFeedForward == strataRoundExpertFF &&
		s.SSMStateSize == strataRoundStateSize && s.SSMConvKernel == strataRoundConvTaps
}

// strataRoundConversions are the tensors the round reads as bf16 and as f16.
func (r *Runner) strataRoundConversions() (bf16, f16 []gguf.TensorInfo, err error) {
	add := func(into *[]gguf.TensorInfo, name string) {
		if info, found := r.file.Tensor(name); found {
			*into = append(*into, info)
		} else if err == nil {
			err = fmt.Errorf("strata round: %s is missing", name)
		}
	}
	for layer := range r.spec.BlockCount {
		prefix := fmt.Sprintf("blk.%d.", layer)
		for _, name := range []string{"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "hc_attn_inject.weight", "hc_ffn_inject.weight"} {
			add(&bf16, prefix+name)
		}
		if !r.qsaLayers[layer] {
			add(&bf16, prefix+"ssm_alpha.weight")
			add(&bf16, prefix+"ssm_beta.weight")
		}
	}
	lexical := fmt.Sprintf("blk.%d.", r.spec.NGram.Layer)
	add(&bf16, lexical+"ple_value.weight")
	add(&f16, lexical+"ple_conv1d.weight")
	return bf16, f16, err
}

// openStrataRound makes the round's stores and CPU pool, once.
func (r *Runner) openStrataRound(ctx context.Context) (*strataRound, error) {
	if r.strata != nil {
		return r.strata, nil
	}
	bf16Tensors, f16Tensors, err := r.strataRoundConversions()
	if err != nil {
		return nil, err
	}
	round := &strataRound{}
	if round.bf16, err = modeldevice.NewDeviceConvertedWeights(r.worker, dtype.BF16); err == nil {
		round.f16, err = modeldevice.NewDeviceConvertedWeights(r.worker, dtype.F16)
	}
	if err == nil {
		loader := modeldevice.NewLoader(r.file)
		loader.Converted(round.bf16, bf16Tensors)
		loader.Converted(round.f16, f16Tensors)
		err = loader.Load(ctx, nil)
	}
	if err != nil {
		return nil, errors.Join(err, round.close(context.WithoutCancel(ctx), r.cuda))
	}
	round.cpu = hostexpert.NewCorePool()
	r.strata = round
	return round, nil
}

// close releases the round.
func (s *strataRound) close(ctx context.Context, cuda *executor.Executor) error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.round != nil {
		errs = append(errs, cuda.CloseStrataRound(ctx, s.round))
	}
	if s.drafter != nil {
		errs = append(errs, cuda.CloseStrataDrafter(ctx, s.drafter))
	}
	if s.promptLast != nil {
		errs = append(errs, s.promptLast.Release(ctx))
	}
	for _, store := range []*modeldevice.DeviceConvertedWeights{s.bf16, s.f16} {
		if store != nil {
			errs = append(errs, store.Close())
		}
	}
	if s.cpu != nil {
		s.cpu.Close()
	}
	return errors.Join(errs...)
}

// strataRoundWeights reads the round's weights by name from the runner's stores.
type strataRoundWeights struct {
	r     *Runner
	round *strataRound
	err   error
}

func (w *strataRoundWeights) fail(name string) driver.DevicePtr {
	if w.err == nil {
		w.err = fmt.Errorf("strata round: %s is not on the card", name)
	}
	return 0
}

// raw is a tensor as the file stores it.
func (w *strataRoundWeights) raw(name string) driver.DevicePtr {
	if tensor, found := w.r.rawWeights.Lookup(name); found {
		return tensor.Pointer
	}
	return w.fail(name)
}

// native is a projection as stored, with its GGML type.
func (w *strataRoundWeights) native(name string) executor.StrataRoundWeight {
	tensor, found := w.r.rawWeights.Lookup(name)
	if !found {
		w.fail(name)
		return executor.StrataRoundWeight{}
	}
	return executor.StrataRoundWeight{Pointer: tensor.Pointer, Type: int(tensor.Info.Type)}
}

// f32 is an f32 tensor.
func (w *strataRoundWeights) f32(name string) driver.DevicePtr {
	if tensor, found := w.r.deviceWeights.Lookup(name); found {
		return tensor.Pointer
	}
	return w.fail(name)
}

// bf16 is a tensor the round reads as bf16.
func (w *strataRoundWeights) bf16(name string) driver.DevicePtr {
	if tensor, found := w.round.bf16.Lookup(name); found {
		return tensor.Pointer
	}
	return w.fail(name)
}

// f16 is a tensor the round reads as f16.
func (w *strataRoundWeights) f16(name string) driver.DevicePtr {
	if tensor, found := w.round.f16.Lookup(name); found {
		return tensor.Pointer
	}
	return w.fail(name)
}

// hyper is a hyper-connection read's weights, prefix "blk.N.hc_attn_" or "output_hc_".
func (w *strataRoundWeights) hyper(prefix string, inject bool) executor.StrataRoundHyper {
	hyper := executor.StrataRoundHyper{Norm: w.f32(prefix + "norm.weight"), Q8Down: w.raw(prefix + "down.weight"), Q8Up: w.raw(prefix + "up.weight")}
	if inject {
		hyper.Inject = w.bf16(prefix + "inject.weight")
	}
	return hyper
}

// strataRoundSpecOf is the round's spec over the sequence's state in past: the weights, the live state and cells, the
// experts' residency.
func (r *Runner) strataRoundSpecOf(round *strataRound, past *deviceKVCache) (executor.StrataRoundSpec, []modeldevice.ExpertBlobLayer, error) {
	cells := past.storage.qsa
	live := cells.live
	w := &strataRoundWeights{r: r, round: round}
	s := r.spec
	resident, experts := r.rawWeights.ExpertResidentTable()
	if resident == 0 || experts != strataRoundExperts {
		return executor.StrataRoundSpec{}, nil, errors.New("strata round: the experts' residency holds no table at Flash-Next's experts")
	}
	spec := executor.StrataRoundSpec{
		Embedding: w.raw("token_embd.weight"), EmbeddingRow: strataRoundEmbedBytes,
		Head: w.native("output.weight"), HeadHyper: w.hyper("output_hc_", false), Vocab: int64(s.VocabularySize),
		SharedFF: int64(s.SharedExpertFF), ConvChannels: int64(s.SSMInnerSize) + tensor.PairedExtent*int64(s.SSMGroupCount)*int64(s.SSMStateSize),
		ValueDim: int64(s.SSMInnerSize), ValueHeads: int64(s.SSMTimeStepRank), KeyHeads: int64(s.SSMGroupCount),
		MaxTokens: strataRoundMaxTokens, Positions: cells.positions.Pointer, Resident: resident, ExpertFF: strataRoundExpertFF,
		RopeFreqBase: tensor.QSARopeBase,
	}
	lexical := fmt.Sprintf("blk.%d.", s.NGram.Layer)
	spec.PLE = executor.StrataRoundPLE{
		Key: w.native(lexical + "ple_key.weight"), Value: w.bf16(lexical + "ple_value.weight"),
		NormKey: w.f32(lexical + "ple_norm_key.weight"), NormQuery: w.f32(lexical + "ple_norm_query.weight"),
		NormConv: w.f32(lexical + "ple_norm_conv.weight"), ConvF16: w.f16(lexical + "ple_conv1d.weight"),
	}
	var blobs []modeldevice.ExpertBlobLayer
	for layer := range int(s.BlockCount) {
		prefix := fmt.Sprintf("blk.%d.", layer)
		l := executor.StrataRoundLayer{
			QSA: r.qsaLayers[layer], Attention: w.hyper(prefix+"hc_attn_", true), FFN: w.hyper(prefix+"hc_ffn_", true),
			Router: w.bf16(prefix + "ffn_gate_inp.weight"), SharedGateInput: w.bf16(prefix + "ffn_gate_inp_shexp.weight"),
			SharedGate: w.native(prefix + "ffn_gate_shexp.weight"), SharedUp: w.native(prefix + "ffn_up_shexp.weight"),
			SharedDown: w.native(prefix + "ffn_down_shexp.weight"),
		}
		blob, found := r.rawWeights.ExpertBlobLayer(layer)
		if !found {
			return executor.StrataRoundSpec{}, nil, fmt.Errorf("strata round: layer %d's experts are not one blob", layer)
		}
		parts := blob.Parts()
		if len(parts) != len(expertBlobPartNames) || parts[strataBlobUp].Offset != parts[strataBlobGate].Bytes ||
			parts[strataBlobDown].Offset != tensor.PairedExtent*parts[strataBlobGate].Bytes {
			return executor.StrataRoundSpec{}, nil, fmt.Errorf("strata round: layer %d's expert blob is not Strata's layout", layer)
		}
		l.ExpertGateUp, l.ExpertDown = parts[strataBlobGate].Type, parts[strataBlobDown].Type
		blobs = append(blobs, blob)
		if l.QSA {
			c := cells.layers[layer]
			l.Query, l.Key = w.native(prefix+"attn_q.weight"), w.native(prefix+"attn_k.weight")
			l.Value, l.Output = w.native(prefix+"attn_v.weight"), w.native(prefix+"attn_output.weight")
			l.IndexerK, l.IndexerQ = w.raw(prefix+"indexer.k_proj.weight"), w.raw(prefix+"indexer.q_proj.weight")
			l.QueryNorm, l.KeyNorm = w.f32(prefix+"attn_q_norm.weight"), w.f32(prefix+"attn_k_norm.weight")
			l.IndexerQNorm, l.IndexerKNorm = w.f32(prefix+"indexer.q_norm.weight"), w.f32(prefix+"indexer.k_norm.weight")
			l.Cells = executor.StrataRoundQSAState{
				KeyCodes: c.keyCodes.Pointer, ValueCodes: c.valueCodes.Pointer, KeyScales: c.keyScales.Pointer,
				ValueScales: c.valueScales.Pointer, PageTable: cells.pageTable.Pointer, Tail: live.keys[layer].Pointer,
				Spare: live.values[layer].Pointer, Pooled: c.pooled.Pointer, BlockPosition: c.blockPosition.Pointer,
				MaxCells: int64(cells.capacity), Slots: int64(cells.slots),
			}
			if cells.slots > 0 {
				l.Cells.PageTable, l.Cells.StreamMap = c.pageTable.Pointer, c.streamMap.Pointer
				l.Cells.HostKeyCodes, l.Cells.HostValueCodes = c.host[0].Pointer, c.host[1].Pointer
				l.Cells.HostKeyScales, l.Cells.HostValueScales = c.host[tensor.PairedExtent].Pointer, c.host[qsaHostRegions-1].Pointer
			}
		} else {
			l.QKV, l.AttnGate, l.SSMOut = w.native(prefix+"attn_qkv.weight"), w.native(prefix+"attn_gate.weight"), w.native(prefix+"ssm_out.weight")
			l.SSMAlpha, l.SSMBeta = w.bf16(prefix+"ssm_alpha.weight"), w.bf16(prefix+"ssm_beta.weight")
			l.SSMConv, l.SSMNorm = w.f32(prefix+"ssm_conv1d.weight"), w.f32(prefix+"ssm_norm.weight")
			l.SSMDT, l.SSMA = w.f32(prefix+"ssm_dt.bias"), w.f32(prefix+"ssm_a")
			l.History, l.State = live.keys[layer].Pointer, live.values[layer].Pointer
		}
		spec.Layers = append(spec.Layers, l)
	}
	return spec, blobs, w.err
}

// strataRoundSame reports whether two specs name the same addresses, so a round made over one serves the other.
func strataRoundSame(a, b executor.StrataRoundSpec) bool {
	if len(a.Layers) != len(b.Layers) || a.PLE != b.PLE || a.Embedding != b.Embedding || a.Head != b.Head ||
		a.HeadHyper != b.HeadHyper || a.Positions != b.Positions || a.Resident != b.Resident {
		return false
	}
	for index := range a.Layers {
		if a.Layers[index] != b.Layers[index] {
			return false
		}
	}
	return true
}

// beginStrataRound readies the round over past's live state: its verifier made over the spec (again where the
// sequence's addresses moved), its pool over the layers' experts, the PLE history and the n-gram token history handed
// over from the prompt read. The rows of first, the token the first window reads, are read beside the refill, as
// Strata's drafter reads a token's rows once it knows the token: read before the first window, its first layer
// waited for the drive.
func (r *Runner) beginStrataRound(ctx context.Context, past *deviceKVCache, first int32) (*strataRound, error) {
	round, err := r.openStrataRound(ctx)
	if err != nil {
		return nil, err
	}
	spec, blobs, err := r.strataRoundSpecOf(round, past)
	if err != nil {
		return nil, err
	}
	if round.round == nil || !strataRoundSame(round.spec, spec) {
		if round.round != nil {
			if err := r.cuda.CloseStrataRound(ctx, round.round); err != nil {
				return nil, err
			}
			round.round = nil
		}
		if round.round, err = r.cuda.NewStrataRound(ctx, spec); err != nil {
			return nil, err
		}
		round.spec = spec
	}
	lexical := func(tokens []int32, rows []float32) error { return r.strataRoundLexical(ctx, round, tokens, rows) }
	previous, err := r.deviceNGramHistory(ctx, past)
	if err != nil {
		return nil, err
	}
	round.previous = round.previous[:0]
	for _, token := range previous {
		round.previous = append(round.previous, uint32(token))
	}
	if err := r.strataRoundPrefetch(ctx, round, []int32{first}); err != nil {
		return nil, err
	}
	if err := r.openStrataTier(ctx, round, blobs); err != nil {
		return nil, err
	}
	if err := r.strataRefill(ctx, round); err != nil {
		return nil, err
	}
	share := round.share
	if r.strataTestShare != nil {
		share = *r.strataTestShare
	}
	if round.pool, err = newStrataRoundPool(blobs, round.cpu, strataRoundPCIeNum(share), lexical); err != nil {
		return nil, err
	}
	// the prompt's windows are not the decode's
	round.pool.usage = round.usage
	round.pcie.mark(0, 0)
	history, err := r.strataRoundConvolution(ctx, past)
	if err != nil {
		return nil, err
	}
	if err := r.cuda.LoadStrataRoundHistory(ctx, round.round, history); err != nil {
		return nil, err
	}
	return round, nil
}

// strataRoundPCIeShare is the PCIe share of the misses Strata starts a native pack at on a link of 20 GB/s or more
// (generate.cpp: pcie_frac 0.55), in proportion below, before its PcieAuto refines it from the CPU pool's speed.
const strataRoundPCIeShare = 0.55

// strataRoundPCIeNum is pcie_frac's 256ths, rounded (drive.d.pcie_num).
func strataRoundPCIeNum(share float64) int {
	const parts = 1 << strataPoolShareShift
	return max(0, min(parts, int(math.Round(share*parts))))
}

// strataRoundConvolution reads the PLE convolution's history past holds, token-major.
func (r *Runner) strataRoundConvolution(ctx context.Context, past *deviceKVCache) ([]float32, error) {
	layer := int(r.spec.NGram.Layer)
	state, found := past.States[layer][model.CacheStateNGramConvolution]
	if !found {
		return nil, errors.New("strata round: the PLE convolution's history is missing")
	}
	values, err := r.cuda.ReadDeviceValues(ctx, []executor.DeviceValue{state.Value})
	if err != nil {
		return nil, err
	}
	return values[0], nil
}

// strataRoundLexical gathers a window's n-gram embedding rows into rows (ss.ple.table->gather_batch over ngram_rows),
// from the history before the window.
func (r *Runner) strataRoundLexical(ctx context.Context, round *strataRound, tokens []int32, rows []float32) error {
	ids := make([]uint32, len(tokens))
	for index, token := range tokens {
		ids[index] = uint32(token)
	}
	selected, _, err := r.spec.NGramRows(ids, round.previous)
	if err != nil {
		return err
	}
	if err := r.openNGramRows(); err != nil {
		return err
	}
	value, err := r.ngramRows.Load(ctx, selected)
	if err != nil {
		return err
	}
	if len(value.Data) != len(rows) {
		return errors.New("strata round: the n-gram rows differ from the window's")
	}
	copy(rows, value.Data)
	return nil
}

// strataRoundPrefetch starts reading the n-gram rows of a window the reader lacks before the window launches, as
// Strata's verifier prefetches them (prefetch_rows), so its first layer's gather finds them read.
func (r *Runner) strataRoundPrefetch(ctx context.Context, round *strataRound, tokens []int32) error {
	ids := make([]uint32, len(tokens))
	for index, token := range tokens {
		ids[index] = uint32(token)
	}
	selected, _, err := r.spec.NGramRows(ids, round.previous)
	if err != nil {
		return err
	}
	if err := r.openNGramRows(); err != nil {
		return err
	}
	r.ngramRows.Prefetch(ctx, selected)
	return nil
}

// openNGramRows opens the n-gram table's row reader at its first use.
func (r *Runner) openNGramRows() error {
	if r.ngramRows != nil {
		return nil
	}
	reader, err := model.NewHostRows(r.file, *r.weights.NGramTokenEmbedding, model.NGramCacheRows)
	if err == nil {
		r.ngramRows = reader
	}
	return err
}

// advance moves the n-gram token history past the tokens a window kept (commit's ple_prev).
func (s *strataRound) advance(r *Runner, kept []int32) error {
	ids := make([]uint32, len(kept))
	for index, token := range kept {
		ids[index] = uint32(token)
	}
	_, next, err := r.spec.NGramRows(ids, s.previous)
	if err == nil {
		s.previous = next
	}
	return err
}

// endStrataRound hands the sequence's state back: the PLE convolution's history written to past's state token-major,
// and the cache that names the live state after the tokens the round kept, pending its last token.
func (r *Runner) endStrataRound(ctx context.Context, round *strataRound, past *deviceKVCache, kept []tokenizer.TokenID,
	pending tokenizer.TokenID) (*deviceKVCache, error) {
	history, err := r.cuda.StrataRoundHistory(ctx, round.round)
	if err != nil {
		return nil, err
	}
	layer := int(r.spec.NGram.Layer)
	state := past.States[layer][model.CacheStateNGramConvolution]
	if err := r.cuda.WriteDeviceValues(ctx, []executor.DeviceValue{state.Value}, [][]float32{history}); err != nil {
		return nil, err
	}
	previous := make([]float32, len(round.previous))
	for index, token := range round.previous {
		previous[index] = float32(token)
	}
	if historyState, found := past.States[layer][model.CacheStateNGramHistory]; found {
		if err := r.cuda.WriteDeviceValues(ctx, []executor.DeviceValue{historyState.Value}, [][]float32{previous}); err != nil {
			return nil, err
		}
	}
	cells := past.storage.qsa
	cells.live.tokens = past.Tokens + uint32(len(kept))
	if !past.storage.retain() {
		return nil, errors.New("strata round: the sequence's storage is released")
	}
	next := &deviceKVCache{
		storage: past.storage, Keys: append([]executor.DeviceValue(nil), past.Keys...),
		Values: append([]executor.DeviceValue(nil), past.Values...), States: make([]deviceLayerStates, len(past.States)),
		Tokens: past.Tokens + uint32(len(kept)), Position: past.Position + uint32(len(kept)), Selected: pending, ngramHistory: previous,
	}
	for index, states := range past.States {
		if len(states) != 0 {
			next.States[index] = states.Clone()
		}
	}
	if err := rebuildDeviceCachePages(next, r.program.Decode.Session); err != nil {
		return nil, errors.Join(err, next.Release(context.Background()))
	}
	return next, nil
}
