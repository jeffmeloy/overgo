package strataspec

// DraftPolicy chooses, a verify round at a time, the MTP's window or a lookup (suffix) window. The suffix drafter
// proposes the tokens that followed an earlier repeat of the context; taken whenever it proposes more than the MTP, it
// lost 2-8% on ordinary text (a long lookup window costs much more to verify than the MTP's usual 3-4 tokens). The
// policy learns both sides online and compares expected committed tokens per millisecond:
//
//	MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//	lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//	        match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by the margin. It only chooses which drafts to
// verify: the output is unchanged.
type DraftPolicy struct {
	maxT              int
	margin            float64
	cost, costN       [windowMaxT + 1]float64 // round ms by window size
	mtpTokens, mtpN   [windowMaxT + 1]float64 // tokens committed by MTP windows of that size
	ok, bad           [matchBuckets]float64   // lookup drafts accepted / windows cut short, decayed
	chainOK, chainBad [matchBuckets]float64   // chained lookup drafts, the same (reached rounds only)
	rounds            float64                 // rounds observed so far
	last              [windowMaxT + 1]float64 // the round at which each size was last measured
}

// The policy's constants (draft_policy.cpp).
const (
	// windowMaxT is the widest window the policy prices, its tokens.
	windowMaxT = 8
	// matchBuckets is the number of match-length buckets a lookup draft's acceptance is counted in.
	matchBuckets = 4
	// DefaultMargin is the lookup window's required advantage.
	DefaultMargin = 0.03
	costAlpha     = 0.1  // EMA weight of a new round time
	tokenAlpha    = 0.05 // EMA weight of a new MTP window outcome
	decay         = 0.97 // lookup counts: older windows fade
	priorN        = 4.0
	probes        = 3     // a lookup window size is tried this often before its guessed cost can veto it
	staleRounds   = 300.0 // a size not measured for this many rounds is tried again (#1252)
	weightCap     = 20.0  // a measured size's most weight in scaling an unmeasured one
	// typicalAcceptance is an MTP window's accepted share before one of its size is seen; confident is the lookup rate,
	// and chainConfident the chain's, past which an unmeasured or stale size is probed.
	typicalAcceptance = 0.7
	confident         = 0.85
	chainConfident    = 0.6
	// the match-length buckets' upper bounds
	bucketShort, bucketMedium, bucketLong = 6, 12, 24
)

// A round's cost by window size relative to one token's (draft_policy.cpp's shape table), between the RTX 5070's
// measured curves: two tokens through eight.
const (
	shapeTwo, shapeThree, shapeFour, shapeFive = 1.35, 1.7, 2.05, 2.45
	shapeSix, shapeSeven, shapeEight           = 2.85, 3.25, 3.6
	priorShort, priorMedium, priorLong         = 0.75, 0.88, 0.93
	priorLongest                               = 0.96
	chainPriorShort, chainPriorMedium          = 0.45, 0.65
	chainPriorLong, chainPriorLongest          = 0.8, 0.9
)

// shape is a round's cost by window size relative to one token, for sizes not measured yet (the measured round times
// replace it).
var shape = [windowMaxT + 1]float64{0, 1, shapeTwo, shapeThree, shapeFour, shapeFive, shapeSix, shapeSeven, shapeEight}

// priorQ is a bucket's acceptance before it has data: the longer the match, the likelier its continuation; worth
// priorN observations. chainPrior is a chained draft's, weaker evidence than a plain match of that length.
var (
	priorQ     = [matchBuckets]float64{priorShort, priorMedium, priorLong, priorLongest}
	chainPrior = [matchBuckets]float64{chainPriorShort, chainPriorMedium, chainPriorLong, chainPriorLongest}
)

// NewDraftPolicy is DraftPolicy(max_t, margin).
func NewDraftPolicy(maxT int, margin float64) *DraftPolicy {
	return &DraftPolicy{maxT: min(max(maxT, 1), windowMaxT), margin: margin}
}

// Pick is a round's window: a lookup's or the MTP's, of T tokens (1 + drafts).
type Pick struct {
	Lookup bool
	T      int
}

func bucket(match int) int {
	for b, bound := range [...]int{bucketShort, bucketMedium, bucketLong} {
		if match < bound {
			return b
		}
	}
	return matchBuckets - 1
}

// LookupRate is the current q for a match length.
func (p *DraftPolicy) LookupRate(match int) float64 {
	b := bucket(match)
	return (p.ok[b] + priorN*priorQ[b]) / (p.ok[b] + p.bad[b] + priorN)
}

// CostMS is the measured or scaled round time of a window of t tokens.
func (p *DraftPolicy) CostMS(t int) float64 {
	t = min(max(t, 1), windowMaxT)
	if p.costN[t] > 0 {
		return p.cost[t]
	}
	// scale from the measured sizes, weighting each by how often it was seen
	var num, den float64
	for u := 1; u <= windowMaxT; u++ {
		if p.costN[u] > 0 {
			w := min(p.costN[u], weightCap)
			num += w * p.cost[u] * shape[t] / shape[u]
			den += w
		}
	}
	if den > 0 {
		return num / den
	}
	return shape[t]
}

func (p *DraftPolicy) stale(t int) bool { return p.costN[t] > 0 && p.rounds-p.last[t] > staleRounds }

func (p *DraftPolicy) mtpTokensOf(t int) float64 {
	if p.mtpN[t] > 0 {
		return p.mtpTokens[t]
	}
	return 1 + typicalAcceptance*float64(t-1) // before any MTP window of this size: a typical acceptance
}

// Choose is the round's window: tMTP the MTP's window, lookupK the lookup proposal's length (0: none), match its
// match length.
func (p *DraftPolicy) Choose(tMTP, lookupK, match int) Pick {
	pick := Pick{T: min(max(tMTP, 1), p.maxT)}
	if lookupK <= 0 {
		return pick
	}
	base := p.mtpTokensOf(pick.T) / p.CostMS(pick.T)
	q := p.LookupRate(match)
	e, qi, best := float64(1), float64(1), float64(0)
	bestT := 0
	for k := 1; k <= min(lookupK, p.maxT-1); k++ {
		qi *= q
		e += qi
		if r := e / p.CostMS(k+1); r > best {
			best, bestT = r, k+1
		}
	}
	full := min(lookupK, p.maxT-1) + 1
	if bestT > 0 && best > base*(1+p.margin) {
		pick.Lookup, pick.T = true, bestT
		// a wider size whose cost has gone stale is re-measured, so one slow stretch cannot price it out for good
		if bestT < full && q >= confident && p.stale(full) {
			pick.T = full
		}
		return pick
	}
	// a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would need
	// a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
	if full > pick.T && (p.costN[full] < probes || p.stale(full)) && q >= confident {
		pick.Lookup, pick.T = true, full
	}
	return pick
}

// Observe records a round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
func (p *DraftPolicy) Observe(lookup bool, t, accepted, match int, roundMS float64) {
	t = min(max(t, 1), windowMaxT)
	p.measure(t, roundMS)
	if lookup {
		b := bucket(match)
		p.ok[b] = decay*p.ok[b] + float64(accepted)
		p.bad[b] = decay * p.bad[b]
		if accepted < t-1 {
			p.bad[b]++
		}
		return
	}
	p.observeMTP(t, accepted)
}

// measure folds a round's time into its size's cost and counts the round.
func (p *DraftPolicy) measure(t int, roundMS float64) {
	if roundMS > 0 {
		if p.costN[t] > 0 {
			p.cost[t] = (1-costAlpha)*p.cost[t] + costAlpha*roundMS
		} else {
			p.cost[t] = roundMS
		}
		p.costN[t]++
		p.last[t] = p.rounds
	}
	p.rounds++
}

// observeMTP folds an MTP window's committed tokens into its size's mean.
func (p *DraftPolicy) observeMTP(t, accepted int) {
	got := float64(accepted) + 1
	if p.mtpN[t] > 0 {
		p.mtpTokens[t] = (1-tokenAlpha)*p.mtpTokens[t] + tokenAlpha*got
	} else {
		p.mtpTokens[t] = got
	}
	p.mtpN[t]++
}

// ChainRate is a chained draft's acceptance for a match length.
func (p *DraftPolicy) ChainRate(match int) float64 {
	b := bucket(match)
	return (p.chainOK[b] + priorN*chainPrior[b]) / (p.chainOK[b] + p.chainBad[b] + priorN)
}

// Chain is --lookup-chain's choice: how many of kAvail lookup tokens to append after an MTP window of tMTP tokens
// whose drafts all hold with probability pMTP; 0 for none.
func (p *DraftPolicy) Chain(tMTP int, pMTP float64, kAvail, match int) int {
	tMTP = min(max(tMTP, 1), p.maxT)
	kMax := min(kAvail, p.maxT-tMTP)
	if kMax <= 0 {
		return 0
	}
	e0 := p.mtpTokensOf(tMTP)
	base, c := e0/p.CostMS(tMTP), p.ChainRate(match)
	pMTP = min(max(pMTP, 0), 1)
	gain, ci, best := float64(0), float64(1), float64(0)
	bestK := 0
	for k := 1; k <= kMax; k++ {
		ci *= c
		gain += ci
		if r := (e0 + pMTP*gain) / p.CostMS(tMTP+k); r > best {
			best, bestK = r, k
		}
	}
	full := tMTP + kMax
	if bestK > 0 && best > base*(1+p.margin) {
		// As Choose: a profitable shorter chain must not keep a stale full size from being measured again.
		if bestK < kMax && pMTP*c >= chainConfident && p.stale(full) {
			return kMax
		}
		return bestK
	}
	// as Choose: a size whose cost is only guessed is tried a few times when the continuation looks likely
	if (p.costN[full] < probes || p.stale(full)) && pMTP*c >= chainConfident {
		return kMax
	}
	return 0
}
