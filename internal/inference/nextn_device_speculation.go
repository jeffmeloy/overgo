package inference

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// deviceSpeculationDraftTokens: NextN self-drafting depth per verify span.
// Audited on the 27B across draft depths two, three, and four: three wins
// on both a predictable-text regime (28.5 tokens per second, 74-78 percent
// chain acceptance) and open-ended prose (17.2, about 40 percent); two
// under-drafts and four over-pays for decayed tail acceptance: it trades a
// deeper chain's accepted drafts against its decayed tail's verify. Provisional:
// the load-costs step derives it from the acceptance measured by depth.
const deviceSpeculationDraftTokens = 3

// hyperDraftTokens: a hyper-connection draft layer's depth per verify span.
// On Flash-Next's short chat a draft from a fresh session holds 63 percent
// at depth one and 43 and 27 percent at depths two and three, so deeper
// spans pay more in verify and draft time than their accepted tokens return:
// it trades a span's accepted drafts against its verify and draft time. The
// window step's chained drafts replace it.
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

// generateDeviceSpeculative: the NextN MTP generation loop. Each round
// drafts up to speculationDraftTokens successors with the draft head,
// then verifies the pending emitted tokens plus the draft in ONE device span
// forward — one trunk weight read that commits acceptedTokens+1 emissions.
// The boundary cache is the recurrent-state checkpoint: a fully accepted
// span becomes the next boundary, a rejected suffix is rolled back by
// releasing the span cache and re-spanning from the boundary.
//   - a lazy grammar awaiting its trigger takes each token, and the loop ends at the token that triggers it, the
//     cache it answers then holding every id before the ones still pending; sampled reports that ending, after which
//     the grammar chooses the tokens
//   - a sampled request (drawing) verifies by Strata's rule (src/core/verify.cpp run, sampler.cu): the window's rows
//     publish their logits and are drawn in order after the window, each at its own position, and a draft is kept while
//     it equals its row's draw; the first row that differs gives the correction, and no row past it is drawn, so the
//     sampler moves as plain sampling moves and a seed gives the text plain decoding gives
func (r *Runner) generateDeviceSpeculative(
	ctx context.Context,
	ids []tokenizer.TokenID,
	prefill *deviceKVCache,
	options GenerateOptions,
	drawing bool,
) (_ []tokenizer.TokenID, _ *deviceKVCache, sampled bool, _ string, _ error) {
	var generatedText strings.Builder
	emitted := 0
	stopped, ended, triggered := false, false, false
	lazy := options.Sampler.GreedyUntilTrigger()
	deliver := func(event TokenEvent) error {
		event.Index = emitted
		ids = append(ids, event.ID)
		emitted++
		stop, err := r.deliverGenerationToken(&event, options, &generatedText)
		if err != nil {
			return err
		}
		ended = stop || emitted >= options.MaxNewTokens
		stopped = ended || triggered
		return nil
	}
	emit := func(id tokenizer.TokenID) error {
		if lazy {
			if err := options.Sampler.AcceptToken(int(id)); err != nil {
				return err
			}
			triggered = !options.Sampler.GreedyUntilTrigger()
		}
		return deliver(TokenEvent{ID: id})
	}
	// draw samples a row, its logits or the card's top K of them, after the ids emitted so far, at the position of the
	// last: the sampler's draw is keyed by it.
	draw := func(logits []float32, candidates []LogitCandidate) (TokenEvent, error) {
		return r.sampleGenerationRow(logits, candidates, ids, options)
	}
	boundary := prefill
	first := boundary.Selected
	if drawing {
		event, err := draw(boundary.Logits, boundary.Candidates)
		if err == nil {
			err = deliver(event)
		}
		if err != nil {
			return ids, boundary, false, "", err
		}
		first = event.ID
	} else if err := emit(first); err != nil {
		return ids, boundary, false, "", err
	}
	pending := []tokenizer.TokenID{first}
	// A QSA runner's windows read its sequence's live state, into which the prompt read's state moves first.
	if r.windowsDecode() {
		live, err := r.liveDeviceCache(ctx, boundary)
		if err != nil {
			return ids, boundary, false, "", err
		}
		if live != boundary && !r.ownsDevicePromptCache(boundary) {
			if err := boundary.Release(ctx); err != nil {
				return ids, boundary, false, "", errors.Join(err, live.Release(context.Background()))
			}
		}
		boundary = live
	}
	depth := r.speculationDraftTokens()
	width := uint64(r.spec.EmbeddingLength)
	if r.hyperDraft != nil {
		width = r.hyperDraftWidth()
	}
	var seedHidden reference.Value
	var seedPosition uint32
	haveSeed := false
	// A hyper draft's round reads the last span's kept rows, each with the token after it, into its history (seedRows
	// from the row at seedStart, seedTokens their next tokens), and drafts from the last.
	var seedRows reference.Value
	var seedTokens []tokenizer.TokenID
	var seedStart uint32
	if prompt := r.hyperDraft.promptRead(); prompt != nil && prompt.position+1 == boundary.Tokens {
		// The prompt's cells are in the history: the first round writes its last row's, with the token it selected, and
		// drafts from it.
		seedRows, seedTokens, seedStart, haveSeed = prompt.last, []tokenizer.TokenID{boundary.Selected}, prompt.position, true
	} else if r.hyperDraft != nil {
		cells := min(options.MaxNewTokens, int(r.spec.ContextLength)) + int(q8InputSpanColumnsWindow) + 1
		if err := r.ensureHyperDraftHistory(ctx, uint32(cells)); err != nil {
			return ids, boundary, false, "", err
		}
	}
	// deferred is the last window's commit, which runs after the next draft.
	var deferred func() error
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
		if haveSeed && room > 0 && len(pending)+depth <= int(q8InputSpanColumnsWindow) && (depth > 1 || len(pending) == 1) && r.hyperDraft != nil {
			drafted, draftErr := r.draftHyperFromHistoryLocked(ctx, seedRows, seedTokens, seedStart)
			if draftErr != nil {
				return ids, boundary, false, "", draftErr
			}
			drafts = append(drafts, drafted)
		} else if haveSeed && room > 0 && len(pending)+depth <= int(q8InputSpanColumnsWindow) && (depth > 1 || len(pending) == 1) {
			session, sessionErr := r.newDeviceMTPSessionLocked(seedHidden, seedPosition)
			if sessionErr != nil {
				return ids, boundary, false, "", sessionErr
			}
			token := pending[len(pending)-1]
			for len(drafts) < room {
				drafted, next, draftErr := r.draftDeviceTokenLocked(ctx, token, session)
				if draftErr != nil {
					return ids, boundary, false, "", draftErr
				}
				session = next
				token = drafted
				drafts = append(drafts, token)
			}
		}
		if deferred != nil {
			if err := deferred(); err != nil {
				return ids, boundary, false, "", err
			}
			deferred = nil
		}
		spanTokens := append(append([]tokenizer.TokenID(nil), pending...), drafts...)
		// A window with drafts keeps its layers' state after its pending tokens and after each prefix of its drafts
		// but the whole, so a window that keeps part of them commits to it without reading them again.
		checkpoints := make([]uint32, len(drafts))
		for index := range checkpoints {
			checkpoints[index] = uint32(len(pending) + index)
		}
		// A window keeps every row in the head: greedy, each row's selection; sampled, each row's logits, or the card's top
		// K of them while the sampler cuts to the top K first (a tool request's, until its grammar triggers).
		spanMode, spanTopK := deviceOutputGreedySpan, uint32(0)
		if drawing {
			if spanMode, spanTopK = deviceOutputLogitsSpan, r.sampledTopK(options); spanTopK != 0 {
				spanMode = deviceOutputTopKSpan
			}
		}
		spanPlan, err := compileDeviceOutputPlan(spanMode, spanTopK, r.spec.VocabularySize)
		if err != nil {
			return ids, boundary, false, "", err
		}
		var span *deviceKVCache
		if r.windowsDecode() {
			// Strata's verifier: the window reads the live state where it sits; its commit keeps what it accepts.
			span, err = r.forwardDeviceWindowLocked(ctx, spanPlan, spanTokens, boundary)
		} else {
			var next []*deviceKVCache
			appended := deviceBatchAppend{Tokens: spanTokens, Past: boundary, Checkpoints: checkpoints}
			if next, err = r.forwardDeviceCachedBatchLocked(ctx, []deviceBatchAppend{appended}, spanPlan); err == nil {
				span = next[0]
			}
		}
		if err != nil {
			return ids, boundary, false, "", err
		}
		base := len(pending) - 1
		accepted := 0
		var correction tokenizer.TokenID
		if drawing {
			if len(span.SpanLogits)+len(span.SpanCandidates) != len(spanTokens) {
				return ids, boundary, false, "", errors.Join(errors.New("inference: span logits are incomplete"), span.Release(context.Background()))
			}
			// Each row is drawn after the ones before it, at its own position; a draft is kept while it equals its row's
			// draw, and the first row that differs, or the last, gives the correction.
			for row := base; !stopped; row++ {
				event, drawErr := draw(span.spanRow(row))
				if drawErr == nil {
					drawErr = deliver(event)
				}
				if drawErr != nil {
					return ids, boundary, false, "", errors.Join(drawErr, span.Release(context.Background()))
				}
				// A draw that triggers the grammar ends the window: the rows after it came down as their top K, which
				// the grammar now masks; the drawn token is the correction the next window reads.
				if spanTopK != 0 && r.sampledTopK(options) == 0 {
					correction = event.ID
					break
				}
				if offset := row - base; offset < len(drafts) && event.ID == drafts[offset] {
					accepted++
					continue
				}
				correction = event.ID
				break
			}
			if options.OnDraftSpan != nil && len(drafts) > 0 {
				options.OnDraftSpan(len(drafts), accepted)
			}
		} else {
			selections := span.SpanSelected
			if len(selections) != len(spanTokens) {
				releaseErr := span.Release(context.Background())
				return ids, boundary, false, "", errors.Join(
					errors.New("inference: span selections are incomplete"), releaseErr,
				)
			}
			for accepted < len(drafts) && drafts[accepted] == selections[base+accepted] {
				accepted++
			}
			if options.OnDraftSpan != nil && len(drafts) > 0 {
				options.OnDraftSpan(len(drafts), accepted)
			}
			correction = selections[base+accepted]
			for index := 0; index < accepted && !stopped; index++ {
				if err := emit(drafts[index]); err != nil {
					return ids, boundary, false, "", errors.Join(err, span.Release(context.Background()))
				}
				if triggered && !ended {
					// The span keeps its state before the draft that triggered the grammar, which is all that is pending.
					accepted, correction = index, drafts[index]
				}
			}
			if !stopped {
				if err := emit(correction); err != nil {
					return ids, boundary, false, "", errors.Join(err, span.Release(context.Background()))
				}
			}
		}
		spanStart := span.Position - uint32(len(spanTokens))
		seedHidden, err = spanHiddenRow(span.SpanHidden, width, base+accepted)
		if err != nil {
			return ids, boundary, false, "", errors.Join(err, span.Release(context.Background()))
		}
		seedPosition = spanStart + uint32(base+accepted) + 1
		haveSeed = true
		if r.hyperDraft != nil {
			// The kept rows are the span's up to the correction's, each followed by the span's next token, the last by the
			// correction.
			kept := base + accepted + 1
			seedRows = reference.Value{Shape: tensor.MustShape(width, uint64(kept)), Data: slices.Clone(span.SpanHidden.Data[:uint64(kept)*width])}
			seedTokens = append(slices.Clone(spanTokens[1:kept]), correction)
			seedStart = spanStart
		}
		// A window whose drafts were all accepted is the next boundary; one that rejected a draft keeps the span at its
		// checkpoint after the pending tokens and the drafts accepted, so the correction is all that is pending.
		committed := accepted == len(drafts)
		if r.windowsDecode() {
			// verify.cpp commit: the window's pending token and accepted drafts advance the live state; a one-token
			// window advanced it itself.
			kept := len(pending) + accepted
			if len(spanTokens) > tensor.SingletonExtent {
				// The commit waits for the next draft, which reads only the window's rows and its own cache, so the
				// experts' advance runs behind the draft as before; the next window, or the request's end, runs it first.
				cells, pastTokens, tokens := boundary.storage.qsa, boundary.Tokens, len(spanTokens)
				deferred = func() error { return r.commitWindowLocked(ctx, cells, pastTokens, tokens, kept) }
			}
			if err := r.followNGramHistory(ctx, span, boundary, spanTokens[:kept]); err != nil {
				return ids, boundary, false, "", errors.Join(err, span.Release(context.Background()))
			}
			span.Tokens, span.Position = boundary.Tokens+uint32(kept), boundary.Position+uint32(kept)
			span.Selection = executor.DeviceValue{}
			committed = true
		} else if !committed && len(span.checkpoints) != 0 {
			kept := append(slices.Clone(pending), drafts[:accepted]...)
			if err := r.commitSpanCheckpoint(ctx, span, boundary, kept); err != nil {
				return ids, boundary, false, "", errors.Join(err, span.Release(context.Background()))
			}
			committed = true
		}
		if committed {
			previous := boundary
			boundary = span
			// A boundary the prompt cache holds stays its entry.
			if !r.ownsDevicePromptCache(previous) {
				if err := previous.Release(ctx); err != nil {
					return ids, boundary, false, "", err
				}
			}
			pending = []tokenizer.TokenID{correction}
		} else {
			if err := span.Release(ctx); err != nil {
				return ids, boundary, false, "", err
			}
			pending = append(append(pending, drafts[:accepted]...), correction)
		}
	}
	if deferred != nil {
		if err := deferred(); err != nil {
			return ids, boundary, false, "", err
		}
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return ids, boundary, false, "", err
	}
	return ids, boundary, triggered && !ended, text, nil
}
