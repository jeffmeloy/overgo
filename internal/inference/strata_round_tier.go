package inference

import (
	"cmp"
	"context"
	"math"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/modeldevice"
)

// generate.cpp's adaptive tier and PcieAuto around the round (Strata at 296b59d2, MIT), at the bench's defaults: the
// VRAM tier follows the conversation - every 4 decode windows (--adapt-every) the host experts routed at least twice
// swap in for the least routed card experts they beat by 1.5 routes (--adapt-min-gain), at most 96 swaps a time
// (--adapt-swaps), the route counts falling to 0.7 of themselves after each adaptation (--adapt-decay), the swaps'
// copies landing before the next window reads the table; and the PCIe share starts from the link's probe and is refined
// from the CPU pool's speed in the first decode windows.

// The adaptive tier's settings.
const (
	strataAdaptEvery    = 4
	strataAdaptSwaps    = 96
	strataAdaptMinUses  = 2.0
	strataAdaptMinGain  = 1.5
	strataAdaptDecay    = 0.7
	strataPCIeLinkFull  = 20.0 // the GB/s from which the native pack's 0.55 stands (pcie_frac_for_gbps)
	strataPCIeFactor    = 1.85 // PcieAuto's kPoolFactor
	strataPCIeFirst     = 48   // the windows before the first refinement, then strataPCIeLater between the next two
	strataPCIeLater     = 96
	strataPCIeMinJobs   = 300  // too few missed experts to read the pool's speed below this
	strataPCIeGiveUp    = 2000 // the windows after which a pool that never misses stops being read
	strataPCIeUpdates   = 3
	strataPCIeMinShare  = 0.10
	strataPCIeMaxShare  = 0.60
	strataPCIeSteps     = 20.0  // the share in steps of 0.05
	strataPCIeStill     = 0.025 // a change below this leaves the share
	strataPCIeMinMicros = 1.0
	strataMicrosecond   = 1e-6
	strataMicrosPerMS   = 1000.0
	strataGBytes        = 1e9
)

// strataPCIeAuto is PcieAuto: the link's rate and the experts' mean blob, and the pool's run time and missed experts read
// since the last mark, accumulated over the windows since the last refinement.
type strataPCIeAuto struct {
	on                bool
	link, blob        float64
	markMS, accMS     float64
	markJobs, accJobs int64
	windows, updates  int
}

// mark notes the pool's counters where the next window's reading starts.
func (p *strataPCIeAuto) mark(runMS float64, jobs int64) { p.markMS, p.markJobs = runMS, jobs }

// window reads a decode window's counters: the new share, or false while there is nothing to change.
func (p *strataPCIeAuto) window(runMS float64, jobs int64, current float64) (float64, bool) {
	if !p.on {
		return 0, false
	}
	p.accMS += runMS - p.markMS
	p.accJobs += jobs - p.markJobs
	p.mark(runMS, jobs)
	p.windows++
	if p.windows < strataPCIeFirst || p.updates != 0 && p.windows < strataPCIeLater {
		return 0, false
	}
	if p.accJobs < strataPCIeMinJobs { // everything hits: look again
		if p.windows >= strataPCIeGiveUp {
			p.on = false
		}
		return 0, false
	}
	micros := p.accMS * strataMicrosPerMS / float64(p.accJobs)
	p.accMS, p.accJobs, p.windows = 0, 0, 0
	if p.updates++; p.updates >= strataPCIeUpdates {
		p.on = false
	}
	if !(micros > strataPCIeMinMicros) || !(p.blob > 0) || !(p.link > 0) {
		return 0, false
	}
	rate := p.blob / (micros * strataMicrosecond) / strataGBytes
	share := min(strataPCIeMaxShare, max(strataPCIeMinShare, math.Round(p.link/(p.link+strataPCIeFactor*rate)*strataPCIeSteps)/strataPCIeSteps))
	if math.Abs(share-current) < strataPCIeStill {
		return 0, false
	}
	return share, true
}

// openStrataTier readies the round's tier once: the route counts, and the PCIe share from the link's probe (the native
// pack's 0.55 from 20 GB/s up, in proportion below), refined by PcieAuto unless a check fixes the share.
func (r *Runner) openStrataTier(ctx context.Context, round *strataRound, layers []modeldevice.ExpertBlobLayer) error {
	if round.usage != nil {
		return nil
	}
	experts := 0
	var blob float64
	for _, layer := range layers {
		experts = layer.Experts()
		blob += float64(layer.Bytes())
	}
	round.share = strataRoundPCIeShare
	if r.strataTestShare == nil {
		link, err := r.cuda.ProbeLinkGBps(ctx)
		if err != nil {
			return err
		}
		if link > 0 {
			round.share = strataRoundPCIeShare * min(1, link/strataPCIeLinkFull)
			round.pcie = strataPCIeAuto{on: true, link: link, blob: blob / float64(max(1, len(layers)))}
		}
	}
	round.usage = make([]float32, len(layers)*experts)
	return nil
}

// strataAdapt is adapt: unless swaps are still in flight, each layer's host experts routed at least twice, the most
// routed first, against its card experts, the least routed first, pair by pair while the host expert's routes beat the
// card expert's by the gain; the largest gains across the layers, at most the swaps a time, queued; then the counts
// decay.
func (r *Runner) strataAdapt(ctx context.Context, round *strataRound) error {
	if round.swapping {
		return nil
	}
	type routed struct {
		uses   float32
		expert int
	}
	type swap struct {
		gain           float32
		layer, in, out int
	}
	var swaps []swap
	var candidates, victims []routed
	for l, layer := range round.pool.layers {
		experts := layer.Experts()
		uses := round.usage[l*experts : (l+1)*experts]
		candidates, victims = candidates[:0], victims[:0]
		for e := range experts {
			if _, onCard := layer.Blob(e); !onCard {
				if uses[e] >= strataAdaptMinUses {
					candidates = append(candidates, routed{uses[e], e})
				}
			} else {
				victims = append(victims, routed{uses[e], e})
			}
		}
		if len(candidates) == 0 || len(victims) == 0 {
			continue
		}
		slices.SortStableFunc(candidates, func(a, b routed) int { return cmp.Compare(b.uses, a.uses) })
		slices.SortStableFunc(victims, func(a, b routed) int { return cmp.Compare(a.uses, b.uses) })
		for i := range min(len(candidates), len(victims)) {
			if candidates[i].uses < victims[i].uses+strataAdaptMinGain {
				break
			}
			swaps = append(swaps, swap{candidates[i].uses - victims[i].uses, l, candidates[i].expert, victims[i].expert})
		}
	}
	slices.SortStableFunc(swaps, func(a, b swap) int { return cmp.Compare(b.gain, a.gain) })
	swaps = swaps[:min(len(swaps), strataAdaptSwaps)]
	if len(swaps) != 0 {
		moves := make([]modeldevice.ExpertSwap, len(swaps))
		for index, s := range swaps {
			moves[index] = modeldevice.ExpertSwap{Layer: s.layer, In: s.in, Out: s.out}
		}
		var queued int
		if err := r.worker.Do(ctx, func(state *device.State) error {
			var err error
			queued, err = r.rawWeights.SwapExperts(ctx, state, moves)
			return err
		}); err != nil {
			return err
		}
		round.swapping = queued > 0
	}
	for index := range round.usage {
		round.usage[index] *= strataAdaptDecay
	}
	return nil
}

// strataRefill is the prompt path's refill after its loan (generate.cpp's refill): the experts the prompt read's
// reclaims moved to host memory return to the card in the memory lent back, their copies queued and waited for at once,
// before the request's first token as before Strata's first window, so the windows see the whole expert cache.
func (r *Runner) strataRefill(ctx context.Context, round *strataRound) error {
	var queued int
	if err := r.worker.Do(ctx, func(state *device.State) error {
		var err error
		queued, err = r.rawWeights.RefillExperts(ctx, state)
		return err
	}); err != nil {
		return err
	}
	if queued == 0 {
		return nil
	}
	round.swapping = true
	return r.strataLand(ctx, round)
}

// strataLand is apply_pending(true): the swaps' copies waited for, their experts on the card, the tables published.
func (r *Runner) strataLand(ctx context.Context, round *strataRound) error {
	if !round.swapping {
		return nil
	}
	round.swapping = false
	return r.worker.Do(ctx, func(state *device.State) error { return r.rawWeights.LandExperts(ctx, state) })
}

// strataPCIeStep is pcie_auto_step after a decode window: the share refined from the pool's speed, the pool taking it.
func (s *strataRound) strataPCIeStep() {
	pool := s.pool
	share, changed := s.pcie.window(float64(pool.runTime)/float64(strataNanosPerMS), pool.misses, s.share)
	if !changed {
		return
	}
	s.share = share
	pool.pcieNum = strataRoundPCIeNum(share)
}

// strataNanosPerMS is a millisecond's nanoseconds.
const strataNanosPerMS = 1e6
