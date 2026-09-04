// Package longform verifies a model's long-input, long-output behaviour
// before a suite pass spends hours on it (owner rule 2026-09-04): one
// long prompt, one greedy generation, the prompt and decode rates the
// run measured, and two degeneration measures over the output. The
// verdict is committed as evidence and the suite passes read it back.
package longform

import (
	"encoding/binary"
)

// DistinctNGramRatio reports the distinct n-grams of the token sequence
// as a fraction of all its n-grams. A sequence shorter than one n-gram
// has nothing to repeat and reports 1; a generation locked in a loop of
// period p reports about p over the n-gram count.
func DistinctNGramRatio(tokens []int32, n int) float64 {
	if n <= 0 || len(tokens) < n {
		return 1
	}
	total := len(tokens) - n + 1
	seen := make(map[string]struct{}, total)
	key := make([]byte, 4*n)
	for start := range total {
		for offset, token := range tokens[start : start+n] {
			binary.LittleEndian.PutUint32(key[4*offset:], uint32(token))
		}
		seen[string(key)] = struct{}{}
	}
	return float64(len(seen)) / float64(total)
}

// LongestRepeatedSpan reports the length of the longest token span that
// occurs at two distinct positions of the sequence; overlapping
// occurrences count, so a period-p loop reports the loop's whole run
// minus p. The quadratic scan is exact and the outputs it reads are a
// few hundred tokens.
func LongestRepeatedSpan(tokens []int32) int {
	longest := 0
	// row[j] holds the common-prefix length of the suffixes at i and j
	// for the row's i; j walks upward so row[j+1] still carries the
	// previous i's value (the prefix length at i+1, j+1) when row[j] is
	// computed, and row[len] stays zero as the end of the sequence.
	row := make([]int, len(tokens)+1)
	for i := len(tokens) - 1; i >= 0; i-- {
		for j := i + 1; j < len(tokens); j++ {
			if tokens[i] == tokens[j] {
				row[j] = row[j+1] + 1
				longest = max(longest, row[j])
			} else {
				row[j] = 0
			}
		}
	}
	return longest
}
