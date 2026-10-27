package strataspec

import "testing"

// ObserveChain records a chained round (--lookup-chain, which the serve loop does not run): the MTP window's size, the
// chained tokens, the window's accepted drafts, its time.
func (p *DraftPolicy) ObserveChain(tMTP, k, accepted, match int, roundMS float64) {
	p.measure(min(max(tMTP+k, 1), windowMaxT), roundMS)
	// the MTP part is an ordinary MTP window as far as its own drafts go
	p.observeMTP(tMTP, min(accepted, tMTP-1))
	if accepted < tMTP-1 {
		return // the chain was never reached: it says nothing about the lookup
	}
	b, ok := bucket(match), accepted-(tMTP-1)
	p.chainOK[b] = decay*p.chainOK[b] + float64(ok)
	p.chainBad[b] = decay * p.chainBad[b]
	if ok < k {
		p.chainBad[b]++
	}
}

// policyCost is draft_policy_test's measured round cost (ms) by window size on the RTX 5070: ~10 ms more a token.
func policyCost(t int) float64 { return 19.0 + 10.5*float64(t-1) }

// TestDraftPolicy is Strata's draft_policy_test (src/spec/draft_policy_test.cpp).
func TestDraftPolicy(t *testing.T) {
	t.Run("no proposal keeps the MTP window", func(t *testing.T) {
		p := NewDraftPolicy(6, DefaultMargin)
		for range 50 {
			p.Observe(false, 4, 2, 0, policyCost(4))
		}
		if k := p.Choose(4, 0, 0); k.Lookup || k.T != 4 {
			t.Fatalf("picked %+v", k)
		}
	})
	t.Run("rejected lookups stop, accepted ones are taken, buckets apart", func(t *testing.T) {
		p := NewDraftPolicy(6, DefaultMargin)
		for range 50 {
			p.Observe(false, 4, 2, 0, policyCost(4))
		}
		for size := 2; size <= 6; size++ {
			p.Observe(false, size, 0, 0, policyCost(size))
		}
		for range 40 {
			p.Observe(true, 6, 0, 4, policyCost(6)) // short matches, all rejected
		}
		if p.LookupRate(4) >= 0.15 {
			t.Errorf("rejected short-match drafts' rate %v is not below 0.15", p.LookupRate(4))
		}
		if p.Choose(4, 5, 4).Lookup {
			t.Error("rejected short-match drafts are still taken")
		}
		for range 40 {
			p.Observe(true, 6, 5, 30, policyCost(6)) // long matches, all accepted
		}
		if p.LookupRate(30) <= 0.9 {
			t.Errorf("accepted long-match drafts' rate %v is not above 0.9", p.LookupRate(30))
		}
		if k := p.Choose(4, 5, 30); !k.Lookup || k.T != 6 {
			t.Errorf("accepted long matches pick %+v, not the full lookup window", k)
		}
		if p.Choose(4, 5, 4).Lookup {
			t.Error("buckets are not separate: short matches are taken")
		}
		if p.Choose(4, 20, 30).T > 6 {
			t.Error("a window beyond the cap")
		}
	})
	t.Run("a mediocre lookup does not replace a strong MTP window", func(t *testing.T) {
		p := NewDraftPolicy(8, DefaultMargin)
		for range 50 {
			p.Observe(false, 3, 2, 0, policyCost(3))
		}
		for size := 2; size <= 8; size++ {
			p.Observe(false, size, size-1, 0, policyCost(size))
		}
		for range 40 {
			p.Observe(true, 4, 2, 8, policyCost(4))
		}
		if p.Choose(3, 7, 8).Lookup {
			t.Error("the mediocre lookup is taken")
		}
	})
	t.Run("an unmeasured size is probed, then its cost decides", func(t *testing.T) {
		p := NewDraftPolicy(6, DefaultMargin)
		for range 50 {
			p.Observe(false, 4, 3, 0, policyCost(4))
		}
		if k := p.Choose(4, 5, 40); !k.Lookup || k.T != 6 {
			t.Errorf("a confident lookup's unmeasured size is not probed: %+v", k)
		}
		for range 3 {
			p.Observe(true, 6, 5, 40, 3.0*policyCost(6))
		}
		if p.Choose(4, 5, 40).Lookup {
			t.Error("after the probes, the measured cost does not decide")
		}
	})
	t.Run("a stale wide size is tried again", func(t *testing.T) {
		p := NewDraftPolicy(6, DefaultMargin)
		for range 50 {
			p.Observe(false, 4, 3, 0, policyCost(4))
		}
		for size := 2; size <= 6; size++ {
			for range 4 {
				p.Observe(false, size, size-1, 0, policyCost(size))
			}
		}
		for range 6 {
			p.Observe(true, 6, 5, 40, 4.0*policyCost(6))
		}
		for range 30 {
			p.Observe(true, 4, 3, 40, policyCost(4))
		}
		saw := false
		for i := 0; i < 400 && !saw; i++ {
			k := p.Choose(4, 5, 40)
			if k.Lookup && k.T == 6 {
				saw = true
				p.Observe(true, 6, 5, 40, policyCost(6))
			} else {
				p.Observe(k.Lookup, k.T, k.T-1, 40, policyCost(k.T))
			}
		}
		if !saw {
			t.Error("a stale wide size is not tried again within a few hundred rounds")
		}
	})
	t.Run("lookup chains", func(t *testing.T) {
		cheap := func(size int) float64 { return 40.0 + 2.0*float64(size-1) }
		p := NewDraftPolicy(8, DefaultMargin)
		for size := 2; size <= 8; size++ {
			for range 5 {
				p.Observe(false, size, 1, 0, cheap(size))
			}
		}
		for range 30 {
			p.Observe(false, 4, 2, 0, cheap(4))
		}
		for range 30 {
			p.ObserveChain(4, 3, 6, 12, cheap(7))
		}
		if k := p.Chain(4, 0.9, 3, 12); k != 3 {
			t.Errorf("cheap rows, always accepted: %d chained, not 3", k)
		}
		q := NewDraftPolicy(8, DefaultMargin)
		for size := 2; size <= 8; size++ {
			for range 5 {
				q.Observe(false, size, 1, 0, policyCost(size))
			}
		}
		for range 30 {
			q.ObserveChain(4, 3, 3, 4, policyCost(7))
		}
		if k := q.Chain(4, 0.9, 3, 4); k != 0 {
			t.Errorf("dear rows, never accepted: %d chained", k)
		}
		if q.Chain(4, 0.9, 0, 4) != 0 || q.Chain(8, 1.0, 3, 40) != 0 {
			t.Error("nothing proposed or no room still chains")
		}
	})
	t.Run("a stale full chain is re-probed behind a profitable shorter one", func(t *testing.T) {
		p := NewDraftPolicy(6, DefaultMargin)
		for range 3 {
			p.Observe(false, 4, 3, 0, 40.0)
		}
		for range 3 {
			p.ObserveChain(4, 1, 4, 40, 42.0)
		}
		for range 3 {
			p.ObserveChain(4, 2, 5, 40, 1000.0)
		}
		if p.Chain(4, 1.0, 2, 40) != 1 {
			t.Fatal("the profitable shorter chain does not win at first")
		}
		probesSeen, lastProbe := 0, -1
		spaced, confidentGate := true, true
		for i := range 1000 {
			k := p.Chain(4, 1.0, 2, 40)
			if k == 2 {
				probesSeen++
				if lastProbe < 0 {
					spaced = spaced && i >= 300
				} else {
					spaced = spaced && i-lastProbe > 300
				}
				lastProbe = i
				confidentGate = confidentGate && p.Chain(4, 0.5, 2, 40) < 2
			}
			cost := 42.0
			if k == 2 {
				cost = 1000.0
			}
			p.ObserveChain(4, k, 3+k, 40, cost)
		}
		if probesSeen == 0 || probesSeen > 3 || !spaced {
			t.Errorf("the stale full size is re-probed %d times, spaced %v", probesSeen, spaced)
		}
		if probesSeen == 0 || !confidentGate {
			t.Error("the stale re-probes do not keep the MTP confidence gate")
		}
	})
}
