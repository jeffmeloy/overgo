package tensorstats

import "math"

// half is the chance of a win where a win and a loss are as likely, the
// share of a miss rate on one side of two, and the step of a bisection.
const half = 0.5

// PairedBinary is two arms compared on independent pairs, each pair a binary
// outcome for either arm, with every count it rests on: a win is a pair the
// first arm passed and the second failed, a loss the reverse, and a tie a pair
// both passed or both failed.
type PairedBinary struct {
	Pairs, Wins, Losses, Ties uint64
	// Tail is the chance, were a discordant pair as likely a win as a loss, of
	// at least Wins wins among the Wins and Losses.
	Tail float64
}

// ExactPairedBinary compares two arms by the exact test for paired binary
// outcomes. The null is that a pair on which the arms differ is as likely a
// win as a loss; ties say nothing of it, and are counted and not used. The
// tail is one-sided and exact, a sum of binomial terms and no approximation
// of one, and is 1 when no pair is discordant: nothing was seen.
//
// The pairs are taken as independent, and the count of them is the caller's
// claim. Several prompts of one admitted group are one pair: counted as
// several they state a smaller tail than the data gives.
func ExactPairedBinary(wins, losses, ties uint64) PairedBinary {
	return PairedBinary{Pairs: wins + losses + ties, Wins: wins, Losses: losses, Ties: ties, Tail: BinomialUpperTail(wins+losses, wins, half)}
}

// GroupedSignTail is the exact one-sided tail of a paired comparison whose pairs come in groups that are not
// independent within themselves, as several questions about one clip: each group is one unit, nets[g] its wins less
// its losses, and the null, that the arms are exchangeable within a group, flips every group's sign at once with chance
// one half. The tail is the chance, under that null, of a total at least the nets' sum, summed exactly over the 2^n sign
// choices by their totals: a group's absolute net is a step either way, and the totals are integers within the steps'
// sum. A group that nets nothing adds nothing, and with no net in any group the tail is 1: nothing was seen. Where
// every group holds one pair, it is ExactPairedBinary's tail; questions that share a clip no longer count as as many
// independent pairs, which stated a smaller tail than the data gives (preset-evidence/statistic, SQA-342).
func GroupedSignTail(nets []int64) float64 {
	var sum, span int64
	for _, net := range nets {
		sum, span = sum+net, span+max(net, -net)
	}
	if span == 0 {
		return 1
	}
	// chance[total+span] is the chance of each total of the groups flipped so far.
	chance := make([]float64, 2*span+1)
	chance[span] = 1
	for _, net := range nets {
		step := max(net, -net)
		if step == 0 {
			continue
		}
		next := make([]float64, len(chance))
		for at, held := range chance {
			if held == 0 {
				continue
			}
			next[int64(at)+step] += held * half
			next[int64(at)-step] += held * half
		}
		chance = next
	}
	tail := float64(0)
	for _, held := range chance[sum+span:] {
		tail += held
	}
	return min(1, tail)
}

// BinomialUpperTail is the chance of at least successes successes in trials
// independent trials of one chance each. It sums the terms from successes up
// and those below it, each from the one before it and in units of the term at
// successes, until a term no longer changes its sum, and divides. No factorial
// is taken, so its error does not grow with the trials, and its cost grows
// with their square root. A tail below the smallest float64 is 0.
func BinomialUpperTail(trials, successes uint64, chance float64) float64 {
	switch {
	case successes == 0 || chance >= 1 && successes <= trials:
		return 1
	case successes > trials || chance <= 0:
		return 0
	}
	n, k, odds := float64(trials), float64(successes), chance/(1-chance)
	above := float64(1)
	for term, at := float64(1), k; at < n; at++ {
		if term *= (n - at) / (at + 1) * odds; above+term == above {
			break
		}
		above += term
	}
	below := float64(0)
	for term, at := float64(1), k; at > 0; at-- {
		if term *= at / (n - at + 1) / odds; below+term == below {
			break
		}
		below += term
	}
	if math.IsInf(above, 1) {
		return 1
	}
	return above / (above + below)
}

// ExactShareBounds bounds the chance of a success from successes successes in
// trials independent trials. Whatever that chance is, it lies below the lower
// bound at most half of miss of the time and above the upper at most half of
// miss: the lower bound is the chance at which so many successes or more have
// that probability, and the upper the chance at which so many or fewer have.
// These are the exact bounds for a share (Clopper and Pearson's), which are
// never short of their rate where a normal approximation's are at small
// counts and at shares near 0 and 1. Each is solved to the last places of a
// float64 from a tail summed in float64, and no trials bound nothing. The
// trials are taken as independent, as ExactPairedBinary's pairs are.
func ExactShareBounds(successes, trials uint64, miss float64) (lower, upper float64) {
	lower, upper = 0, 1
	if successes > 0 {
		lower, _ = firstChance(func(chance float64) bool { return BinomialUpperTail(trials, successes, chance) > miss*half })
	}
	if successes < trials {
		_, upper = firstChance(func(chance float64) bool { return 1-BinomialUpperTail(trials, successes+1, chance) < miss*half })
	}
	return lower, upper
}

// firstChance brackets where a condition that holds for every chance above
// some chance, and for none below it, begins to hold: the last float64 it does
// not hold at and the next, which it does.
func firstChance(holds func(chance float64) bool) (below, at float64) {
	below, at = 0, 1
	for {
		middle := below + (at-below)*half
		if middle == below || middle == at {
			return below, at
		}
		if holds(middle) {
			at = middle
		} else {
			below = middle
		}
	}
}
