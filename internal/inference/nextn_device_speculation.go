package inference

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// deviceSpeculationDraftTokens: NextN self-drafting depth per verify span.
// Audited on the 27B across draft depths two, three, and four: three wins
// on both a predictable-text regime (28.5 tokens per second, 74-78 percent
// chain acceptance) and open-ended prose (17.2, about 40 percent); two
// under-drafts and four over-pays for decayed tail acceptance.
const deviceSpeculationDraftTokens = 3

// hyperDraftTokens: a hyper-connection draft layer's depth per verify span.
// On Flash-Next's short chat a draft from a fresh session holds 63 percent
// at depth one and 43 and 27 percent at depths two and three, so deeper
// spans pay more in verify and draft time than their accepted tokens return.
const hyperDraftTokens = 1

// speculationDraftTokens is the depth the runner's draft program drafts per verify span.
func (r *Runner) speculationDraftTokens() int {
	if r.hyperDraft != nil {
		return hyperDraftTokens
	}
	return deviceSpeculationDraftTokens
}

// q8InputSpanColumnsWindow mirrors the executor's q8-input span-column
// window: a verify span longer than this falls off the single-weight-read
// kernels, so drafting is gated to keep spans inside it.
const q8InputSpanColumnsWindow = 8

// deviceSpeculationReady reports whether raw-greedy device generation can
// ride NextN speculation: a complete single-head draft catalog in the same
// model or an attached hyper-connection draft layer, device-resident
// weights, and no context shifting (spans append monotonically).
func (r *Runner) deviceSpeculationReady(options GenerateOptions) bool {
	if r == nil || !options.SpeculativeDecode || options.ContextShift ||
		!r.hasPreloadedWeights() {
		return false
	}
	if r.hyperDraft != nil || r.validateNextNMTP() == nil {
		return true
	}
	_, catalog, ok := r.lookupSingleHeadMTP()
	return ok && !catalog.MTPOnly
}

// SpeculativeDraftTokens declares the tokens each speculative verify span
// drafts under raw-greedy device decoding, or zero when the model has no
// draft head this runner drafts from or its weights are not device-resident.
func (r *Runner) SpeculativeDraftTokens() int {
	if !r.deviceSpeculationReady(GenerateOptions{SpeculativeDecode: true}) {
		return 0
	}
	return r.speculationDraftTokens()
}

// SpeculatesByDefault reports whether the runner's raw-greedy device decoding speculates unless a request decodes
// plainly: a draft the runner was opened with, which a model's own draft head is not.
func (r *Runner) SpeculatesByDefault() bool {
	return r.hyperDraft != nil && r.SpeculativeDraftTokens() > 0
}

func greedyTokenID(logits []float32) tokenizer.TokenID {
	best := 0
	for index, value := range logits {
		if value > logits[best] {
			best = index
		}
	}
	return tokenizer.TokenID(best)
}

// newDeviceMTPSessionLocked seeds a draft session from a device span hidden
// row: the trunk lives in a device KV cache, so the session carries only the
// pending hidden and its absolute position.
func (r *Runner) newDeviceMTPSessionLocked(
	hidden reference.Value,
	position uint32,
) (*MTPSession, error) {
	targetModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	session := &MTPSession{
		PendingHidden: hidden, MTPStart: position, Position: position,
		targetModel: targetModel, deviceTrunk: true,
	}
	switch {
	case r.hyperDraft != nil:
		// A hyper draft's cell sits at its streams' position, one before the token they selected; the draft step checks the streams.
		if position == 0 {
			return nil, errors.New("inference: hyper-connection draft seed has no position")
		}
		session.MTPStart, session.Position = position-1, position-1
	case r.validateNextNMTP() == nil:
		err = r.validateNextNMTPSession(session)
	default:
		err = r.validateMTPSession(session)
	}
	if err != nil {
		return nil, err
	}
	return session, nil
}

// draftDeviceTokenLocked runs one draft step through whichever draft program
// the runner holds and returns its greedy token.
func (r *Runner) draftDeviceTokenLocked(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
) (tokenizer.TokenID, *MTPSession, error) {
	if r.hyperDraft != nil {
		return r.advanceHyperDraftLocked(ctx, tokenID, session)
	}
	advance := r.advanceMTPLocked
	if r.validateNextNMTP() == nil {
		advance = r.advanceNextNMTPLocked
	}
	logits, next, err := advance(ctx, tokenID, session)
	if err != nil {
		return 0, nil, err
	}
	return greedyTokenID(logits.Data), next, nil
}

// spanHiddenRow extracts one pre-output-norm hidden column from a span
// append as a draft-session seed.
func spanHiddenRow(hidden reference.Value, width uint64, row int) (reference.Value, error) {
	start := uint64(row) * width
	if len(hidden.Data) < int(start+width) {
		return reference.Value{}, errors.New("inference: span hidden row is out of range")
	}
	data := make([]float32, width)
	copy(data, hidden.Data[start:start+width])
	return reference.Value{Shape: tensor.MustShape(width, 1), Data: data}, nil
}

// generateDeviceSpeculativeGreedy: the NextN MTP generation loop. Each round
// drafts up to speculationDraftTokens successors with the draft head,
// then verifies the pending emitted tokens plus the draft in ONE device span
// forward — one trunk weight read that commits acceptedTokens+1 emissions.
// The boundary cache is the recurrent-state checkpoint: a fully accepted
// span becomes the next boundary, a rejected suffix is rolled back by
// releasing the span cache and re-spanning from the boundary.
func (r *Runner) generateDeviceSpeculativeGreedy(
	ctx context.Context,
	ids []tokenizer.TokenID,
	prefill *deviceKVCache,
	options GenerateOptions,
) ([]tokenizer.TokenID, *deviceKVCache, string, error) {
	var generatedText strings.Builder
	emitted := 0
	stopped := false
	emit := func(id tokenizer.TokenID) error {
		event := TokenEvent{ID: id, Index: emitted}
		ids = append(ids, id)
		emitted++
		stop, err := r.deliverGenerationToken(&event, options, &generatedText)
		if err != nil {
			return err
		}
		stopped = stop || emitted >= options.MaxNewTokens
		return nil
	}
	boundary := prefill
	if err := emit(boundary.Selected); err != nil {
		return ids, boundary, "", err
	}
	pending := []tokenizer.TokenID{boundary.Selected}
	depth := r.speculationDraftTokens()
	width := uint64(r.spec.EmbeddingLength)
	if r.hyperDraft != nil {
		width = r.hyperDraftWidth()
	}
	var seedHidden reference.Value
	var seedPosition uint32
	haveSeed := false
	for !stopped {
		var drafts []tokenizer.TokenID
		// A long pending run means the last drafts were rejected past the
		// first; run a draft-less span to collapse it into a new boundary.
		// A one-draft round commits its checkpoint after every rejection,
		// so each span it verifies is two tokens long and replays one of
		// two frames a cache capacity.
		// A round drafts no more than the output and the context have left: its span emits the accepted drafts and
		// one token more and appends its pending tokens and drafts to the boundary, so the request ends where plain
		// decoding ends.
		room := min(depth, options.MaxNewTokens-emitted-1, int(r.spec.ContextLength)-int(boundary.Tokens)-len(pending))
		if haveSeed && room > 0 && len(pending)+depth <= int(q8InputSpanColumnsWindow) && (depth > 1 || len(pending) == 1) {
			session, sessionErr := r.newDeviceMTPSessionLocked(seedHidden, seedPosition)
			if sessionErr != nil {
				return ids, boundary, "", sessionErr
			}
			token := pending[len(pending)-1]
			for len(drafts) < room {
				drafted, next, draftErr := r.draftDeviceTokenLocked(ctx, token, session)
				if draftErr != nil {
					return ids, boundary, "", draftErr
				}
				session = next
				token = drafted
				drafts = append(drafts, token)
			}
		}
		spanTokens := append(append([]tokenizer.TokenID(nil), pending...), drafts...)
		// A window with drafts keeps its layers' state after its pending tokens and after each prefix of its drafts
		// but the whole, so a window that keeps part of them commits to it without reading them again.
		checkpoints := make([]uint32, len(drafts))
		for index := range checkpoints {
			checkpoints[index] = uint32(len(pending) + index)
		}
		span, err := r.forwardDeviceCachedModeLocked(
			ctx, deviceOutputGreedySpan, spanTokens, boundary, checkpoints...,
		)
		if err != nil {
			return ids, boundary, "", err
		}
		selections := span.SpanSelected
		if len(selections) != len(spanTokens) {
			releaseErr := span.Release(context.Background())
			return ids, boundary, "", errors.Join(
				errors.New("inference: span selections are incomplete"), releaseErr,
			)
		}
		base := len(pending) - 1
		accepted := 0
		for accepted < len(drafts) && drafts[accepted] == selections[base+accepted] {
			accepted++
		}
		if options.OnDraftSpan != nil && len(drafts) > 0 {
			options.OnDraftSpan(len(drafts), accepted)
		}
		for index := 0; index < accepted && !stopped; index++ {
			if err := emit(drafts[index]); err != nil {
				return ids, boundary, "", errors.Join(err, span.Release(context.Background()))
			}
		}
		correction := selections[base+accepted]
		if !stopped {
			if err := emit(correction); err != nil {
				return ids, boundary, "", errors.Join(err, span.Release(context.Background()))
			}
		}
		spanStart := span.Position - uint32(len(spanTokens))
		seedHidden, err = spanHiddenRow(span.SpanHidden, width, base+accepted)
		if err != nil {
			return ids, boundary, "", errors.Join(err, span.Release(context.Background()))
		}
		seedPosition = spanStart + uint32(base+accepted) + 1
		haveSeed = true
		// A window whose drafts were all accepted is the next boundary; one that rejected a draft keeps the span at its
		// checkpoint after the pending tokens and the drafts accepted, so the correction is all that is pending.
		committed := accepted == len(drafts)
		if !committed && len(span.checkpoints) != 0 {
			kept := append(slices.Clone(pending), drafts[:accepted]...)
			if err := r.commitSpanCheckpoint(ctx, span, boundary, kept); err != nil {
				return ids, boundary, "", errors.Join(err, span.Release(context.Background()))
			}
			committed = true
		}
		if committed {
			previous := boundary
			boundary = span
			// A boundary the prompt cache holds stays its entry.
			if !r.ownsDevicePromptCache(previous) {
				if err := previous.Release(ctx); err != nil {
					return ids, boundary, "", err
				}
			}
			pending = []tokenizer.TokenID{correction}
		} else {
			if err := span.Release(ctx); err != nil {
				return ids, boundary, "", err
			}
			pending = append(append(pending, drafts[:accepted]...), correction)
		}
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return ids, boundary, "", err
	}
	return ids, boundary, text, nil
}
