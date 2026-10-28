package inference

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"overgo/internal/cuda/executor"
	"overgo/internal/sampling"
	"overgo/internal/strataspec"
	"overgo/internal/tokenizer"
)

// generate.cpp's single-sequence round for a greedy request (Strata at 296b59d2, MIT), as its serve loop runs it: each
// window feeds the pending token and the drafts the draft layer made at or above min_p through the verifier, commits
// the tokens fed up to the last agreed draft, emits the agreed drafts and the verifier's own next token, and the draft
// layer catches up the window's accepted rows and drafts the next window's chain. Plain decoding is the round's
// one-token windows.

// The suffix drafter as the serve loop makes it: matches of 3 (--suffix-draft) to 64 tokens over a history of the
// context and 4,096 more.
const (
	strataSuffixMinMatch = 3
	strataSuffixMaxMatch = 64
	strataSuffixSlack    = 4096
)

// strataPipedTokens is the tokens the loop hands the delivering goroutine ahead of it before it waits for the
// callbacks: a slow client holds the round back past it.
const strataPipedTokens = 256

// strataRoundOn reports whether a request decodes through the round: a greedy request without a grammar, or a sampled
// one whose chain is Strata's sampler's, on a runner at Flash-Next's geometry that asks for it
// (GenerateOptions.StrataRound).
func (r *Runner) strataRoundOn(options GenerateOptions, deviceGreedy bool) bool {
	if !options.StrataRound || !r.strataRoundModel() {
		return false
	}
	if deviceGreedy {
		return true
	}
	_, sampled := strataSamplingOf(options)
	return sampled
}

// strataSamplerList is the widest top_k list Strata's sampled path keeps (kSelMax): a request truncating to more, or
// not at all, samples from more than it holds: the list its kernel holds.
const strataSamplerList = 64

// strataSamplingOf is a sampled request's chain as Strata's sampler runs it (sampler.hpp's SamplerParams: top_k,
// top_p, min_p, min_keep, temperature, the seed), where the request's chain is that one: a positive temperature, a
// top_k of 1 to 64, no stage Strata's chain lacks, and its stages in llama.cpp's order; false otherwise.
func strataSamplingOf(options GenerateOptions) (executor.StrataSampling, bool) {
	sampler := options.Sampler
	if sampler == nil || sampler.IsRawGreedy() || options.PostSamplingProbabilities != 0 {
		return executor.StrataSampling{}, false
	}
	c := sampler.Config()
	if c.Temperature <= 0 || c.DynatempRange != 0 || c.Mirostat != 0 || c.TopK < 1 || c.TopK > strataSamplerList ||
		c.Grammar != nil || c.GBNF != nil && !sampler.AwaitsTrigger() || len(c.LogitBiases) != 0 || c.Infill != nil || c.NoRepeatNgramSize != 0 ||
		!sampling.PenaltiesIdle(c) ||
		c.DryMultiplier != 0 || c.TopNSigma > 0 || c.TypicalP > 0 && c.TypicalP < 1 || c.XTCProbability > 0 {
		return executor.StrataSampling{}, false
	}
	order := []sampling.SamplerStage{sampling.SamplerTopK, sampling.SamplerTopP, sampling.SamplerMinP, sampling.SamplerTemperature}
	if len(c.Samplers) != 0 {
		next := 0
		for _, stage := range c.Samplers {
			at := slices.Index(order, stage)
			if at < 0 && (stage == sampling.SamplerAdaptiveP || stage == sampling.SamplerInfill) || at >= 0 && at < next {
				return executor.StrataSampling{}, false
			}
			next = max(next, at)
		}
		if !slices.Contains(c.Samplers, sampling.SamplerTemperature) {
			return executor.StrataSampling{}, false
		}
	}
	topP := c.TopP
	if topP <= 0 || topP >= 1 {
		topP = 1
	}
	return executor.StrataSampling{
		TopK: int32(c.TopK), TopP: topP, MinP: max(0, c.MinP), Temperature: c.Temperature, MinKeep: int32(max(1, c.MinKeep)),
		PenaltyRepeat: 1, Seed: uint64(c.Seed),
	}, true
}

// strataRoundDrafter reports whether a round request drafts with the round's draft layer: a runner holding the pack,
// a request that neither decodes plainly nor takes a check's drafts.
func (r *Runner) strataRoundDrafter(options GenerateOptions, deviceGreedy bool) bool {
	return r.strataDraft != nil && r.strataRoundOn(options, deviceGreedy) && !options.PlainDecode && r.strataTestDrafts == nil
}

// strataDraftSource drafts up to want tokens after the pending one, the request's ids so far ending with it.
type strataDraftSource func(ids []tokenizer.TokenID, want int) []tokenizer.TokenID

// strataRoundDrafts is the request's draft source in place of the draft layer: none for plain decoding; the runner's
// test source where one is set.
func (r *Runner) strataRoundDrafts(options GenerateOptions) strataDraftSource {
	if options.PlainDecode {
		return nil
	}
	return r.strataTestDrafts
}

// strataRoundWindowSize is the serve loop's window: the pending token and each draft while its probability reaches
// min_p, at most S_mtp tokens.
func strataRoundWindowSize(draft executor.StrataDraft) int {
	size := 1
	for size < strataRoundMTPTokens && size-1 < draft.Count && draft.Probabilities[size-1] >= strataDraftMinP {
		size++
	}
	return size
}

// generateStrataRound decodes ids' continuation from past through the round as generate.cpp's serve loop does: where
// the read stopped before the prompt's last token, the first window holds that token alone (first_window) and gives
// the first token; a prompt the read took whole (projected inputs) gives its first token from the read's last row.
// Then a window a round; the cache it answers names the sequence's state after the tokens the windows fed, the last
// token pending. evaluated, where set, reports the prompt's time once the round is ready for its first window, where
// Strata's prompt time ends (after its refill).
func (r *Runner) generateStrataRound(
	ctx context.Context, ids []tokenizer.TokenID, past *deviceKVCache, options GenerateOptions, evaluated func() error,
) (_ []tokenizer.TokenID, _ *deviceKVCache, triggered bool, _ error) {
	// firstWindow: the read stopped before the prompt's last token, which the first window reads.
	firstWindow := uint64(past.Tokens)+1 == uint64(len(ids))
	sampling, sampled := strataSamplingOf(options)
	// A prompt read whole gives the first token: a sampled request's drawn from the read's row at its position, as its
	// windows' rows are after it.
	var event TokenEvent
	if !firstWindow {
		event = TokenEvent{ID: past.Selected}
		if sampled {
			var err error
			if event, err = r.sampleGenerationRow(past.Logits, past.Candidates, ids, options); err != nil {
				return ids, past, false, err
			}
		}
	}
	live, err := r.liveDeviceCache(ctx, past)
	if err != nil {
		return ids, nil, false, err
	}
	if live != past && !r.ownsDevicePromptCache(past) {
		if err := past.Release(ctx); err != nil {
			return ids, nil, false, errors.Join(err, live.Release(context.Background()))
		}
	}
	// The first window reads the prompt's last token, or the token a prompt read whole gave.
	first := int32(ids[len(ids)-1])
	if !firstWindow {
		first = int32(event.ID)
	}
	round, err := r.beginStrataRound(ctx, live, first)
	if err != nil {
		return ids, live, false, err
	}
	if sampled {
		round.round.SetStrataRoundSampling(&sampling)
	} else {
		round.round.SetStrataRoundSampling(nil)
	}
	var text strings.Builder
	var fed []tokenizer.TokenID
	// A tool request's lazy grammar masks no token until one it accepts triggers it: the round decodes until then, and
	// the token that triggers it ends the round, the grammar choosing the rest.
	lazy := options.Sampler.AwaitsTrigger()

	deliver := func(event *TokenEvent) (bool, error) {
		if lazy {
			if err := options.Sampler.AcceptToken(int(event.ID)); err != nil {
				return false, err
			}
			triggered = !options.Sampler.AwaitsTrigger()
		}
		stop, err := r.deliverGenerationToken(event, options, &text)
		return stop || triggered, err
	}
	generated := 0
	if !firstWindow {
		ids = append(ids, event.ID)
		generated = 1
	}
	source := r.strataRoundDrafts(options)
	// The draft layer: the prompt's last row read takes the token after it, the prompt's last, which the first window
	// holds (mtp.cpp's prefill of a read's rows, each with the token after it), and the first window's rows draft the
	// first chain; a prompt read whole drafts it with draft_first from that row and the first token.
	var draft executor.StrataDraft
	var loopErr error
	drafting := r.strataRoundDrafter(options, !sampled) && round.drafter != nil
	if drafting {
		loopErr = r.cuda.SetStrataDrafterWindow(ctx, round.drafter, round.round.FinalResidual())
		prompt := r.strataPrompt
		if loopErr == nil && prompt != nil && prompt.pending && uint64(prompt.position)+1 == uint64(live.Tokens) {
			if firstWindow {
				loopErr = r.cuda.PrefillStrataDrafter(ctx, round.drafter, round.promptLastRow(), []int32{int32(ids[len(ids)-1])},
					int64(prompt.position))
				prompt.pending = false
			} else {
				draft, loopErr = r.cuda.FirstDraftStrata(ctx, round.drafter, round.promptLastRow(), int32(live.Selected),
					int64(prompt.position), strataDraftMinP)
			}
		}
	}
	// The suffix drafter's history: the prompt and every token emitted; the policy is learned over the runner's life.
	if drafting {
		if r.strataSuffix == nil {
			r.strataSuffix = strataspec.NewSuffixDrafter(strataSuffixMinMatch, strataSuffixMaxMatch,
				int(r.strataRoundContext())+strataSuffixSlack)
			r.strataPolicy = strataspec.NewDraftPolicy(strataRoundMaxTokens, strataspec.DefaultMargin)
		}
		r.strataSuffix.Reset()
		for _, token := range ids {
			r.strataSuffix.Append(int32(token))
		}
	}
	// The prompt's time ends with the round ready for its first window; a prompt read whole gives its first token
	// then, its drafts and the suffix drafter's history made.
	stop := false
	if loopErr == nil && evaluated != nil {
		loopErr = evaluated()
	}
	if loopErr == nil && !firstWindow {
		stop, loopErr = deliver(&event)
	}
	// Where no callback decides the request's stop (no stop sequence, no stop callback), the loop stops at its own end
	// of text, or at its lazy grammar's trigger, which it takes each token to itself, and the tokens go to the callbacks
	// on a goroutine of their own, in order, written out beside the commit, the draft and the next window, as Strata's
	// engine pipes its tokens to its server: a window's tokens had taken 0.1 to 0.5 ms of each window to write out
	// (agent turns, 2026-10-27). A callback's failure stops the loop at its next token.
	var pipe chan TokenEvent
	var piped chan error
	var pipeFailed atomic.Bool
	if loopErr == nil && !stop && options.ShouldStop == nil && len(options.StopSequences) == 0 {
		pipe, piped = make(chan TokenEvent, strataPipedTokens), make(chan error, 1)
		go func() {
			var err error
			for event := range pipe {
				if err == nil {
					if _, err = r.deliverGenerationToken(&event, options, &text); err != nil {
						pipeFailed.Store(true)
					}
				}
			}
			piped <- err
		}()
	}
	// The pool's workers poll for each layer's experts while the round decodes, as Strata's pool keeps a worker spinning
	// on each core: a layer's experts are due within microseconds of its doorbell.
	round.cpu.Spin(true)
	defer round.cpu.Spin(false)
	// The device worker polls for the round's calls the same way, several a window, as Strata's host calls the driver
	// itself.
	if loopErr == nil {
		loopErr = r.worker.Spin(ctx, true)
		defer func() { _ = r.worker.Spin(context.WithoutCancel(ctx), false) }()
	}
	out := make([]int32, strataRoundMaxTokens)
	// adapting is the adaptive tier's answer while its goroutine runs, nil otherwise.
	var adapting chan error
	lookup := make([]int32, strataRoundMaxTokens)
	for loopErr == nil && !stop && generated < options.MaxNewTokens {
		// the first window holds the prompt's last token alone, and teaches the policy nothing (Strata's timed_round)
		first := firstWindow
		firstWindow = false
		window := []int32{int32(ids[len(ids)-1])}
		var drafted []tokenizer.TokenID
		fromLookup, match := false, 0
		if source != nil && !first {
			drafted = source(ids, min(strataRoundMaxTokens-1, options.MaxNewTokens-generated-1))
		} else if drafting && !first {
			// a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
			// when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
			size, proposal := strataRoundWindowSize(draft), draft.Tokens
			k := r.strataSuffix.Propose(strataRoundMaxTokens-1, lookup)
			match = r.strataSuffix.LastMatch()
			if k > 0 && lookup[0] == draft.Tokens[0] {
				if pick := r.strataPolicy.Choose(size, k, match); pick.Lookup {
					size, proposal, fromLookup = pick.T, lookup, true
				}
			}
			drafted = make([]tokenizer.TokenID, 0, strataRoundMaxTokens-1)
			for _, token := range proposal[:min(size, options.MaxNewTokens-generated)-1] {
				drafted = append(drafted, tokenizer.TokenID(token))
			}
		}
		started := time.Now()
		for _, token := range drafted {
			window = append(window, int32(token))
		}
		// the window's n-gram rows are read beside the copies' landing and the window's first layer
		if loopErr = r.strataRoundPrefetch(ctx, round, window); loopErr != nil {
			break
		}
		// the adaptive tier's copies land before the window reads the table
		if loopErr = r.strataLand(ctx, round); loopErr != nil {
			break
		}
		position := int64(live.Tokens) + int64(len(fed))
		if loopErr = r.cuda.RunStrataRound(ctx, round.round, window, position, round.pool, out); loopErr != nil {
			break
		}
		agreed := 0
		for agreed < len(drafted) && int32(drafted[agreed]) == out[agreed] {
			agreed++
		}
		// every 4th window the VRAM tier follows the conversation on a goroutine of its own, its choice and copies beside
		// the commit and the draft, as Strata's adapt thread runs (joined before the next window lands its copies)
		if round.windows++; round.windows%strataAdaptEvery == 0 {
			adapting = make(chan error, 1)
			go func() { adapting <- r.strataAdapt(ctx, round) }()
		}
		if options.OnDraftSpan != nil && len(drafted) > 0 {
			options.OnDraftSpan(len(drafted), agreed)
		}
		// the agreed drafts and the verifier's own next token, until a token stops the request
		kept := 0
		for kept <= agreed && !stop && loopErr == nil && generated < options.MaxNewTokens {
			event = TokenEvent{ID: tokenizer.TokenID(out[kept]), Index: generated}
			ids = append(ids, event.ID)
			generated++
			kept++
			if pipe != nil {
				if lazy {
					if loopErr = options.Sampler.AcceptToken(int(event.ID)); loopErr != nil {
						break
					}
					triggered = !options.Sampler.AwaitsTrigger()
				}
				pipe <- event
				stop = triggered || pipeFailed.Load() || !options.ContinueAfterEOG && r.vocab.IsEOG(event.ID)
			} else {
				stop, loopErr = deliver(&event)
			}
			if drafting {
				r.strataSuffix.Append(int32(event.ID))
			}
		}
		if loopErr != nil || kept == 0 {
			break
		}
		// a window that drafts the next one commits in the draft's first run, as Strata's commit and first draft step run
		// as one; any other commits alone
		drafts := drafting && !stop && generated < options.MaxNewTokens
		var commit executor.StrataCommit
		if drafts {
			commit, loopErr = executor.PrepareStrataRoundCommit(round.round, kept)
		} else {
			loopErr = r.cuda.CommitStrataRound(ctx, round.round, kept)
		}
		if loopErr != nil {
			break
		}
		if loopErr = round.advance(r, window[:kept]); loopErr != nil {
			break
		}
		for _, token := range window[:kept] {
			fed = append(fed, tokenizer.TokenID(token))
		}
		// the draft layer catches up the window's cells up to the last kept row and drafts the next window's chain
		if drafts {
			// the next window's n-gram rows are read as its tokens become known, as Strata's drafter prefetches them
			// (prefetch_ple): the pending token's before the draft, then each draft's as it is read; one that fails reads
			// nothing ahead, the window's own read failing as it would
			ahead := []int32{out[kept-1]}
			_ = r.strataRoundPrefetch(ctx, round, ahead)
			draft, loopErr = r.cuda.DraftStrata(ctx, round.drafter, out[:len(window)], position, kept-1, strataDraftMinP, commit, func(token int32) {
				ahead = append(ahead, token)
				_ = r.strataRoundPrefetch(ctx, round, ahead)
			})
			// the round's window, its accepted drafts and its time (verify, commit and draft) teach the policy
			if loopErr == nil && !first {
				r.strataPolicy.Observe(fromLookup, len(window), agreed, match, float64(time.Since(started))/float64(time.Millisecond))
			}
		}
		// the PCIe share refined from the pool's speed
		round.strataPCIeStep()
		if adapting != nil {
			loopErr = errors.Join(loopErr, <-adapting)
			adapting = nil
		}
	}
	if adapting != nil {
		loopErr = errors.Join(loopErr, <-adapting)
	}
	// every token delivered before the request ends
	if pipe != nil {
		close(pipe)
		loopErr = errors.Join(loopErr, <-piped)
	}
	next, endErr := r.endStrataRound(ctx, round, live, fed, ids[len(ids)-1])
	if endErr == nil && !r.ownsDevicePromptCache(live) {
		endErr = live.Release(ctx)
	}
	return ids, next, triggered, errors.Join(loopErr, endErr)
}
