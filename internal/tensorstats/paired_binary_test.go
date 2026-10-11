package tensorstats

import (
	"math"
	"math/big"
	"testing"
)

// exactUpperTail is the chance of at least successes successes in trials
// trials of chance each, summed in rational arithmetic from the binomial
// coefficients: the count the package's sum is held to, derived apart from it.
func exactUpperTail(trials, successes int64, chance *big.Rat) *big.Rat {
	sum, miss := new(big.Rat), new(big.Rat).Sub(big.NewRat(1, 1), chance)
	for count := max(successes, 0); count <= trials; count++ {
		term := new(big.Rat).SetInt(new(big.Int).Binomial(trials, count))
		for range count {
			term.Mul(term, chance)
		}
		for range trials - count {
			term.Mul(term, miss)
		}
		sum.Add(sum, term)
	}
	return sum
}

// near holds a float64 to an exact value within parts of it, or of one when it is smaller.
func near(got float64, want *big.Rat, parts float64) bool {
	exact, _ := want.Float64()
	return math.Abs(got-exact) <= parts*max(exact, math.SmallestNonzeroFloat64)
}

// TestExactPairedBinaryVerdictAndBounds holds the exact comparison of paired
// binary outcomes and the exact bounds of a share to counts derived apart
// from them, in rational arithmetic.
//
// The comparison: over every table of up to twelve wins and twelve losses,
// with and without ties, the tail is the share of the equally likely
// arrangements of the discordant pairs with at least the wins seen. Ties are
// counted and move nothing; no discordant pair is a tail of 1, all wins the
// smallest tail the pairs allow, all losses 1. The tail at another chance than
// a half is held the same way, since the bounds are solved from it.
//
// The bounds: for every count of successes in sixteen trials, the lower bound
// is where so many successes or more come to be more likely than half the
// miss rate, and the upper where so many or fewer come to be less likely,
// each placed by the exact tail to a part in a million million either side of
// it. Over chances from one in a hundred to
// ninety-nine, the bounds of what could be seen hold the chance at least as
// often as they state. No successes and all successes have the bounds their
// closed forms give, at sixteen trials and at a thousand million.
//
// At large counts: the tail at half the pairs is half, and half the middle
// term, which Stirling's series gives; a tail and the tail of its mirror are
// the whole; and a table of a million pairs is bounded as a normal
// approximation bounds it, to the fifth place.
func TestExactPairedBinaryVerdictAndBounds(t *testing.T) {
	t.Parallel()
	const parts = 1e-12
	even := big.NewRat(1, 2)
	for wins := range uint64(13) {
		for losses := range uint64(13) {
			for _, ties := range []uint64{0, 5} {
				got := exactPairedBinary(wins, losses, ties)
				want := exactUpperTail(int64(wins+losses), int64(wins), even)
				if got.Pairs != wins+losses+ties || got.Wins != wins || got.Losses != losses || got.Ties != ties || !near(got.Tail, want, parts) {
					t.Fatalf("%d wins, %d losses and %d ties = %+v, want a tail of %s over its own counts", wins, losses, ties, got, want.FloatString(18))
				}
			}
		}
	}
	for name, table := range map[string]struct {
		wins, losses, ties uint64
		want               float64
	}{
		"no discordant pair": {0, 0, 9, 1},
		"all wins":           {10, 0, 0, 1.0 / 1024},
		"all losses":         {0, 10, 0, 1},
		"as many of each":    {1, 1, 0, 0.75},
	} {
		if got := exactPairedBinary(table.wins, table.losses, table.ties).Tail; got != table.want {
			t.Errorf("%s: a tail of %v, want %v", name, got, table.want)
		}
	}
	for _, chance := range []*big.Rat{big.NewRat(1, 100), big.NewRat(1, 3), big.NewRat(7, 10), big.NewRat(99, 100)} {
		rate, _ := chance.Float64()
		for trials := range int64(31) {
			for successes := range trials + 2 {
				if got, want := binomialUpperTail(uint64(trials), uint64(successes), rate), exactUpperTail(trials, successes, new(big.Rat).SetFloat64(rate)); !near(got, want, parts) {
					t.Fatalf("at least %d of %d at a chance of %s = %v, want %s", successes, trials, chance, got, want.FloatString(18))
				}
			}
		}
	}

	const miss, trials = 0.05, 16
	side := big.NewRat(1, 40)
	rational := func(value float64) *big.Rat { return new(big.Rat).SetFloat64(value) }
	bounds := make([][2]float64, trials+1)
	for successes := range int64(trials + 1) {
		lower, upper := exactShareBounds(uint64(successes), trials, miss)
		bounds[successes] = [2]float64{lower, upper}
		if lower < 0 || upper > 1 || lower > float64(successes)/trials || upper < float64(successes)/trials {
			t.Fatalf("%d of %d is bounded by %v and %v, which do not hold the share seen", successes, trials, lower, upper)
		}
		if successes > 0 {
			if exactUpperTail(trials, successes, rational(lower*(1-parts))).Cmp(side) > 0 || exactUpperTail(trials, successes, rational(lower*(1+parts))).Cmp(side) <= 0 {
				t.Fatalf("the lower bound of %d of %d, %v, is not where so many or more come to have more than half the miss rate", successes, trials, lower)
			}
		} else if lower != 0 {
			t.Fatalf("no successes in %d are bounded below by %v", trials, lower)
		}
		if successes < trials {
			// So many or fewer is the whole less one more or more.
			fewer := func(chance float64) *big.Rat {
				return new(big.Rat).Sub(big.NewRat(1, 1), exactUpperTail(trials, successes+1, rational(chance)))
			}
			if fewer(upper*(1+parts)).Cmp(side) >= 0 || fewer(upper*(1-parts)).Cmp(side) < 0 {
				t.Fatalf("the upper bound of %d of %d, %v, is not where so many or fewer come to have less than half the miss rate", successes, trials, upper)
			}
		} else if upper != 1 {
			t.Fatalf("%d successes in %d are bounded above by %v", successes, trials, upper)
		}
	}
	for percent := int64(1); percent < 100; percent++ {
		chance := big.NewRat(percent, 100)
		rate, _ := chance.Float64()
		held := new(big.Rat)
		for successes := range int64(trials + 1) {
			if bounds[successes][0] <= rate && rate <= bounds[successes][1] {
				exactly := new(big.Rat).Sub(exactUpperTail(trials, successes, chance), exactUpperTail(trials, successes+1, chance))
				held.Add(held, exactly)
			}
		}
		if held.Cmp(big.NewRat(95, 100)) < 0 {
			t.Fatalf("at a chance of %s the bounds of %d trials hold it %s of the time, short of the %v they state", chance, trials, held.FloatString(6), 1-miss)
		}
	}
	for _, count := range []uint64{trials, 1e9} {
		edge := math.Pow(miss/2, 1/float64(count))
		if lower, upper := exactShareBounds(count, count, miss); upper != 1 || math.Abs(lower-edge) > parts {
			t.Errorf("%d successes in %d are bounded by %v and %v, want %v and 1", count, count, lower, upper, edge)
		}
		if lower, upper := exactShareBounds(0, count, miss); lower != 0 || math.Abs(upper-(1-edge)) > parts {
			t.Errorf("no successes in %d are bounded by %v and %v, want 0 and %v", count, lower, upper, 1-edge)
		}
	}
	if lower, upper := exactShareBounds(0, 0, miss); lower != 0 || upper != 1 {
		t.Errorf("no trials are bounded by %v and %v, want nothing bounded", lower, upper)
	}

	const pairs = 1_000_000
	// The middle term of an even count, by Stirling's series to its second term.
	middle := math.Sqrt(2/(math.Pi*pairs)) * (1 - 1.0/(4*pairs))
	if got, want := exactPairedBinary(pairs/2, pairs/2, 0).Tail, 0.5+middle/2; math.Abs(got-want) > parts {
		t.Errorf("half of %d pairs won: a tail of %.15f, want %.15f", pairs, got, want)
	}
	for _, wins := range []uint64{1, pairs / 4, pairs/2 - 700, pairs / 2, pairs/2 + 700, pairs - 1} {
		tail, mirror := binomialUpperTail(pairs, wins, half), binomialUpperTail(pairs, pairs-wins+1, half)
		if math.Abs(tail+mirror-1) > parts || tail < 0 || tail > 1 {
			t.Errorf("at least %d of %d and at least its mirror = %v and %v, which are not the whole", wins, pairs, tail, mirror)
		}
	}
	if far := exactPairedBinary(pairs, 0, 0).Tail; far != 0 {
		t.Errorf("a million wins and no loss: a tail of %v, want one below the smallest float64 read as 0", far)
	}
	lower, upper := exactShareBounds(pairs/2, pairs, miss)
	spread := normalHalfWidth * math.Sqrt(0.25/pairs)
	if math.Abs(lower-(0.5-spread)) > 1e-5 || math.Abs(upper-(0.5+spread)) > 1e-5 || lower >= 0.5 || upper <= 0.5 {
		t.Errorf("half of %d is bounded by %v and %v, want about %v either side of it", pairs, lower, upper, spread)
	}
}

// normalHalfWidth is the standard normal's two-sided quantile at a miss rate of one in twenty.
const normalHalfWidth = 1.959963984540054
