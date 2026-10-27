// Package strataspec is Strata's speculation policy for its decode round (strata-port/decode), ported from Strata at
// 296b59d2 (MIT, licenses/Strata-MIT.txt): the suffix-lookup drafter (src/spec/suffix_drafter.cpp) and the draft
// policy that chooses, a verify round at a time, the MTP's window or a lookup window (src/spec/draft_policy.cpp).
package strataspec

// The suffix drafter's index: a trigram's end positions, WAYS most recent first, in an open-addressing table of twice
// the nominal capacity; matches are extended back up to the maximum and shorter than the minimum propose nothing.
const (
	// trigramWays is the end positions a trigram keeps, most recent first.
	trigramWays   = 4
	trigram       = 3
	minimumMatch  = 3
	defaultMax    = 32
	defaultTokens = 1 << 19
	loadFactor    = 2 // the table's slots a token of nominal capacity: a load factor of at most 0.5
	// mix is splitmix64's finalizer; trigram's key mixes the three tokens with the golden ratio's and another odd
	// constant.
	mixShift  = 33
	mixFirst  = 0xff51afd7ed558ccd
	mixSecond = 0xc4ceb9fe1a85ec53
	keyGolden = 0x9E3779B97F4A7C15
	keyOdd    = 0x632BE59BD9B4E019
)

func mix(x uint64) uint64 {
	x ^= x >> mixShift
	x *= mixFirst
	x ^= x >> mixShift
	x *= mixSecond
	return x ^ (x >> mixShift)
}

// slot is a trigram's entry: its key (the hash plus one, zero empty) and its end positions, most recent first.
type slot struct {
	key uint64
	pos [trigramWays]uint32
	n   uint8
}

// SuffixDrafter proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix:
// when the output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify
// pass accepts many tokens at once. It needs no model and costs microseconds.
type SuffixDrafter struct {
	minMatch, maxMatch int
	history            []int32
	table              []slot
	mask               uint64
	lastMatch          int
}

// NewSuffixDrafter is SuffixDrafter(min_match, max_match, capacity_tokens): the minimum at least three.
func NewSuffixDrafter(minMatch, maxMatch, capacityTokens int) *SuffixDrafter {
	d := &SuffixDrafter{minMatch: max(minimumMatch, minMatch), maxMatch: max(minMatch, maxMatch)}
	capacity := uint64(1)
	for capacity < uint64(capacityTokens)*loadFactor {
		capacity <<= 1
	}
	d.table = make([]slot, capacity)
	d.mask = capacity - 1
	d.history = make([]int32, 0, capacityTokens)
	return d
}

// Reset empties the history and the index.
func (d *SuffixDrafter) Reset() {
	d.history = d.history[:0]
	clear(d.table)
	d.lastMatch = 0
}

func (d *SuffixDrafter) keyAt(end int) uint64 {
	a, b, c := uint64(uint32(d.history[end-trigram+1])), uint64(uint32(d.history[end-1])), uint64(uint32(d.history[end]))
	return mix(a*keyGolden^mix(b+keyOdd)^(c<<1)) | 1 // never 0
}

func (d *SuffixDrafter) findSlot(key uint64, insert bool) *slot {
	for i, probes := key&d.mask, uint64(0); probes <= d.mask; i, probes = (i+1)&d.mask, probes+1 {
		s := &d.table[i]
		if s.key == key {
			return s
		}
		if s.key == 0 {
			if !insert {
				return nil
			}
			s.key = key
			return s
		}
	}
	return nil // table full: the history outgrew its capacity
}

// Append adds tokens to the history (the prompt, then every accepted token).
func (d *SuffixDrafter) Append(tokens ...int32) {
	for _, token := range tokens {
		d.history = append(d.history, token)
		end := len(d.history) - 1
		if end < trigram-1 {
			continue
		}
		s := d.findSlot(d.keyAt(end), true)
		if s == nil {
			continue
		}
		for w := trigramWays - 1; w > 0; w-- {
			s.pos[w] = s.pos[w-1]
		}
		s.pos[0] = uint32(end)
		if s.n < trigramWays {
			s.n++
		}
	}
}

// Propose writes up to maxK proposed next tokens to out and answers how many (0: no match of at least the minimum).
func (d *SuffixDrafter) Propose(maxK int, out []int32) int {
	d.lastMatch = 0
	n := len(d.history)
	if n < trigram+1 || maxK <= 0 {
		return 0
	}
	cur := n - 1
	s := d.findSlot(d.keyAt(cur), false)
	if s == nil {
		return 0
	}
	bestEnd, bestLen := 0, 0
	for w := range int(s.n) {
		p := int(s.pos[w])
		if p >= cur {
			continue // the current suffix itself
		}
		length := 0
		for length < d.maxMatch && length <= p && d.history[p-length] == d.history[cur-length] {
			length++
		}
		if length > bestLen { // most recent first, so ties keep the newer
			bestLen, bestEnd = length, p
		}
	}
	if bestLen < d.minMatch {
		return 0
	}
	d.lastMatch = bestLen
	k := 0
	// The continuation may run into the current suffix (periodic text); reading history up to cur is valid.
	for q := bestEnd + 1; q <= cur && k < maxK; q++ {
		out[k] = d.history[q]
		k++
	}
	return k
}

// LastMatch is the length of the match behind the last proposal (0 if none).
func (d *SuffixDrafter) LastMatch() int { return d.lastMatch }

// Size is the history's length.
func (d *SuffixDrafter) Size() int { return len(d.history) }
