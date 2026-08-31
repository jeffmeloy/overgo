package inference

import (
	"context"
	"errors"
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

// q8InputSpanColumnsWindow mirrors the executor's q8-input span-column
// window: a verify span longer than this falls off the single-weight-read
// kernels, so drafting is gated to keep spans inside it.
const q8InputSpanColumnsWindow = 8

// deviceSpeculationReady reports whether raw-greedy device generation can
// ride NextN speculation: a complete single-head draft catalog in the same
// model, device-resident weights, and no context shifting (spans append
// monotonically).
func (r *Runner) deviceSpeculationReady(options GenerateOptions) bool {
	if r == nil || !options.SpeculativeDecode || options.ContextShift ||
		!r.hasPreloadedWeights() {
		return false
	}
	if r.validateNextNMTP() == nil {
		return true
	}
	_, catalog, ok := r.lookupSingleHeadMTP()
	return ok && !catalog.MTPOnly
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
	if r.validateNextNMTP() == nil {
		err = r.validateNextNMTPSession(session)
	} else {
		err = r.validateMTPSession(session)
	}
	if err != nil {
		return nil, err
	}
	return session, nil
}

// advanceDeviceDraftLocked runs one draft step through whichever single-head
// program the model declares.
func (r *Runner) advanceDeviceDraftLocked(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
) (reference.Value, *MTPSession, error) {
	if r.validateNextNMTP() == nil {
		return r.advanceNextNMTPLocked(ctx, tokenID, session)
	}
	return r.advanceMTPLocked(ctx, tokenID, session)
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
// drafts up to deviceSpeculationDraftTokens successors with the NextN head,
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
		return nil, boundary, "", err
	}
	pending := []tokenizer.TokenID{boundary.Selected}
	width := uint64(r.spec.EmbeddingLength)
	var seedHidden reference.Value
	var seedPosition uint32
	haveSeed := false
	for !stopped {
		var drafts []tokenizer.TokenID
		// A long pending run means the last drafts were rejected twice
		// over; run a draft-less span to collapse it into a new boundary.
		if haveSeed && len(pending)+deviceSpeculationDraftTokens <= int(q8InputSpanColumnsWindow) {
			session, sessionErr := r.newDeviceMTPSessionLocked(seedHidden, seedPosition)
			if sessionErr != nil {
				return nil, boundary, "", sessionErr
			}
			token := pending[len(pending)-1]
			for len(drafts) < deviceSpeculationDraftTokens {
				logits, next, draftErr := r.advanceDeviceDraftLocked(ctx, token, session)
				if draftErr != nil {
					return nil, boundary, "", draftErr
				}
				session = next
				token = greedyTokenID(logits.Data)
				drafts = append(drafts, token)
			}
		}
		spanTokens := append(append([]tokenizer.TokenID(nil), pending...), drafts...)
		span, err := r.forwardDeviceCachedModeLocked(
			ctx, deviceOutputGreedySpan, spanTokens, boundary,
		)
		if err != nil {
			return nil, boundary, "", err
		}
		selections := span.SpanSelected
		if len(selections) != len(spanTokens) {
			releaseErr := span.Release(context.Background())
			return nil, boundary, "", errors.Join(
				errors.New("inference: span selections are incomplete"), releaseErr,
			)
		}
		base := len(pending) - 1
		accepted := 0
		for accepted < len(drafts) && drafts[accepted] == selections[base+accepted] {
			accepted++
		}
		for index := 0; index < accepted && !stopped; index++ {
			if err := emit(drafts[index]); err != nil {
				return nil, boundary, "", errors.Join(err, span.Release(context.Background()))
			}
		}
		correction := selections[base+accepted]
		if !stopped {
			if err := emit(correction); err != nil {
				return nil, boundary, "", errors.Join(err, span.Release(context.Background()))
			}
		}
		spanStart := span.Position - uint32(len(spanTokens))
		seedHidden, err = spanHiddenRow(span.SpanHidden, width, base+accepted)
		if err != nil {
			return nil, boundary, "", errors.Join(err, span.Release(context.Background()))
		}
		seedPosition = spanStart + uint32(base+accepted) + 1
		haveSeed = true
		if accepted == len(drafts) {
			previous := boundary
			boundary = span
			if err := previous.Release(ctx); err != nil {
				return nil, boundary, "", err
			}
			pending = []tokenizer.TokenID{correction}
		} else {
			if err := span.Release(ctx); err != nil {
				return nil, boundary, "", err
			}
			pending = append(append(pending, drafts[:accepted]...), correction)
		}
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return nil, boundary, "", err
	}
	return ids, boundary, text, nil
}
