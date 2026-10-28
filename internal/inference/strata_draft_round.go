package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

// The round's drafter over the pack (executor.StrataDrafter), as generate.cpp's serve loop runs it at the bench's
// configuration: --spec 4 with the suffix drafter on loads the drafter at max_t 6 (mtp_t = o.spec once the suffix
// drafter widens it to 6) and caps its chain at S_mtp - 1 = 3 drafts (set_max_drafts); --mtp-window 32768; --spec-min-p
// 0.5 ends a chain at a draft below it and sizes the next window by the drafts at or above it.
const (
	strataDraftMaxTokens = 6
	strataDraftMaxDrafts = 3
	strataDraftWindow    = 32768
	strataDraftMinP      = 0.5
	// strataDraftWindowLead is load's 64 cells past the window that a prompt's prefill still writes (first_needed).
	strataDraftWindowLead = 64
)

// openStrataDraftFile opens the draft file and places its pack on the card.
func openStrataDraftFile(ctx context.Context, cuda *executor.Executor, path string) (*strataDraftPack, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("inference: open draft %s: %w", path, err)
	}
	pack, err := openStrataDraftPack(ctx, cuda, file)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("inference: draft %s: %w", path, err), file.Close())
	}
	return pack, file.Close()
}

// strataDraftFits reports whether the draft file is one the round's drafter reads: a qwen4exp-mtp layer whose routed
// experts are the Q2_0 banks the pack relays, with its draft vocabulary.
func strataDraftFits(path string) bool {
	file, err := gguf.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	gateUp, gateUpFound := file.Tensor("mtp.layers.0.mlp.experts.gate_up_proj")
	down, downFound := file.Tensor("mtp.layers.0.mlp.experts.down_proj")
	_, vocabulary := file.Tensor(model.HyperDraftVocabulary)
	return model.DescribesHyperDraft(file) && gateUpFound && downFound && gateUp.Type == dtype.Q2_0 && down.Type == dtype.Q2_0 && vocabulary
}

// strataDraftPrompt is a round request's prompt read for the drafter: from, the first position whose row the drafter
// takes; the read's rows' streams; whether the last row read (the round's promptLast) waits at position for the token
// after it (pending).
type strataDraftPrompt struct {
	from, position uint32
	pending        bool
	streams        *executor.DeviceBuffer
}

// strataRoundContext is the cells a round request's sequence holds: the streamed context, or the model's.
func (r *Runner) strataRoundContext() int64 {
	if r.qsaStreamedContext != 0 {
		return int64(r.qsaStreamedContext)
	}
	return int64(r.spec.ContextLength)
}

// strataDrafterSpec is the drafter's spec over the pack, the trunk's embedding and head, and the session's cells.
func (r *Runner) strataDrafterSpec(round *strataRound) (executor.StrataDrafterSpec, error) {
	pack := r.strataDraft
	var errs []error
	q8 := func(name string) executor.StrataRoundWeight {
		pointer, err := pack.weight(name, "q8_0")
		errs = append(errs, err)
		return executor.StrataRoundWeight{Pointer: pointer, Type: int(dtype.Q8_0)}
	}
	get := func(name, kind string) driver.DevicePtr {
		pointer, err := pack.weight(name, kind)
		errs = append(errs, err)
		return pointer
	}
	hyper := func(prefix string, inject bool) executor.StrataRoundHyper {
		h := executor.StrataRoundHyper{
			Norm: get(prefix+".hc_norm.weight", "f32"), Down: get(prefix+".input_mix_weight_down.weight", "bf16"),
			Up: get(prefix+".input_mix_weight_up.weight", "bf16"),
		}
		if inject {
			h.Inject = get(prefix+".block_inject_weight.weight", "bf16")
		}
		return h
	}
	w := &strataRoundWeights{r: r, round: round}
	head, found := r.rawWeights.Lookup("output.weight")
	if !found || head.Info.Shape[1] == 0 {
		return executor.StrataDrafterSpec{}, errors.New("mtp: the draft layer needs the native head")
	}
	spec := executor.StrataDrafterSpec{
		FCEmbedding: q8("fc_embedding.weight"), FCHidden: q8("fc_hidden.weight"), Query: q8("self_attn.q_proj.weight"),
		Key: q8("self_attn.k_proj.weight"), Value: q8("self_attn.v_proj.weight"), Output: q8("self_attn.o_proj.weight"),
		SharedGate: q8("mlp.shared_expert.gate_proj.weight"), SharedUp: q8("mlp.shared_expert.up_proj.weight"),
		SharedDown:    q8("mlp.shared_expert.down_proj.weight"),
		EmbeddingNorm: get("pre_fc_norm_embedding.weight", "f32"), HiddenNorm: get("pre_fc_norm_hidden.weight", "f32"),
		QueryNorm: get("self_attn.q_norm.weight", "f32"), KeyNorm: get("self_attn.k_norm.weight", "f32"),
		Attention: hyper("attn_hyper_connection", true), FFN: hyper("mlp_hyper_connection", true),
		Mixer:  hyper("hyper_connection_mixer", false),
		Router: get("mlp.gate.weight", "bf16"), SharedGateInput: get("mlp.shared_expert_gate.weight", "bf16"),
		Experts: pack.expertsBase, ExpertBlob: int64(strataDraftBlobBytes()),
		MainHead:    executor.StrataRoundWeight{Pointer: head.Pointer, Type: int(head.Info.Type)},
		MainHeadRow: head.Info.Size / head.Info.Shape[1], Vocabulary: pack.vocabulary,
		Embedding: w.raw("token_embd.weight"), EmbeddingRow: strataRoundEmbedBytes,
		SharedFF:  int64(pack.tensors["mlp.shared_expert.gate_proj.weight"].rows),
		MaxTokens: strataDraftMaxTokens, MaxDrafts: strataDraftMaxDrafts, Window: strataDraftWindow,
		MaxCells: r.strataRoundContext(), RopeFreqBase: tensor.QSARopeBase,
	}
	errs = append(errs, w.err)
	return spec, errors.Join(errs...)
}

// openStrataDrafter makes the round's drafter over the pack, once.
func (r *Runner) openStrataDrafter(ctx context.Context) (*strataRound, error) {
	round, err := r.openStrataRound(ctx)
	if err != nil || round.drafter != nil && round.drafterCells == r.strataRoundContext() {
		return round, err
	}
	// a session of another context: the K/V's cells (the ring's page table) are its own
	if round.drafter != nil {
		if err := r.cuda.CloseStrataDrafter(ctx, round.drafter); err != nil {
			return nil, err
		}
		round.drafter = nil
	}
	spec, err := r.strataDrafterSpec(round)
	if err != nil {
		return nil, err
	}
	if round.drafter, err = r.cuda.NewStrataDrafter(ctx, spec); err != nil {
		return nil, err
	}
	round.drafterCells = spec.MaxCells
	return round, nil
}

// beginStrataDraftPromptLocked readies the drafter for a round request that reads its prompt of prompt tokens after
// cached: the prompt's length set (prefill skips the cells the window can never reach again), and the read keeping its
// rows' streams from the window's first position.
func (r *Runner) beginStrataDraftPromptLocked(ctx context.Context, prompt, cached int) error {
	round, err := r.openStrataDrafter(ctx)
	if err != nil {
		return err
	}
	round.drafter.SetStrataDrafterPrompt(int64(prompt))
	r.strataPrompt = &strataDraftPrompt{from: uint32(max(cached, prompt-strataDraftWindow-strataDraftWindowLead))}
	return nil
}

// recordStrataDraftPromptLocked writes into the drafter's K/V the cells a prompt read's rows make, the first row at
// position first, from the read's streams on the card: the last row read before them with this read's first token,
// then each row but the last with the token after it, from the window's first position; the last row is kept for the
// round's first draft.
func (r *Runner) recordStrataDraftPromptLocked(ctx context.Context, streams executor.DeviceValue, kind dtype.Type, tokens []tokenizer.TokenID, first uint32) error {
	prompt, drafter := r.strataPrompt, r.strata.drafter
	width := uint64(strataRoundStreams * strataRoundEmbedding)
	rows := uint64(len(tokens))
	if elements, err := streams.Shape.Elements(); err != nil || kind != dtype.F32 || rows == 0 || streams.Pointer == 0 || elements != rows*width {
		return errors.New("inference: a prompt read's draft streams are incompatible")
	}
	rowBytes := width * strataDraftF32Bytes
	if prompt.pending && prompt.position+1 == first && prompt.position >= prompt.from {
		if err := r.cuda.PrefillStrataDrafter(ctx, drafter, r.strata.promptLastRow(), []int32{int32(tokens[0])}, int64(prompt.position)); err != nil {
			return err
		}
	}
	skip := uint64(min(uint32(rows-1), prompt.from-min(prompt.from, first)))
	next := make([]int32, 0, rows)
	for _, token := range tokens[skip+1:] {
		next = append(next, int32(token))
	}
	if err := r.cuda.PrefillStrataDrafter(ctx, drafter, streams.Pointer+driver.DevicePtr(skip*rowBytes), next, int64(first)+int64(skip)); err != nil {
		return err
	}
	if r.strata.promptLast == nil {
		var err error
		if r.strata.promptLast, r.strata.promptLastBase, err = strataDraftBuffer(ctx, r.cuda, rowBytes); err != nil {
			return err
		}
	}
	if err := r.cuda.CopyDeviceRanges(ctx, []executor.DeviceRangeCopy{{
		Target: r.strata.promptLastRow(), Source: streams.Pointer + driver.DevicePtr((rows-1)*rowBytes), Bytes: rowBytes,
	}}); err != nil {
		return err
	}
	prompt.position, prompt.pending = first+uint32(rows-1), true
	return nil
}

// promptLastRow is the address of the last prompt row read.
func (s *strataRound) promptLastRow() driver.DevicePtr { return s.promptLastBase }
