package tokenizer

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// scanMergeBPE is the independent, deliberately quadratic ranked-merge
// reference. Its equal-rank choice is the first adjacent pair.
func scanMergeBPE(input []string, lookup func(string, string) (int, bool)) []string {
	parts := slices.Clone(input)
	for len(parts) > 1 {
		best := -1
		var bestRank int
		for i := 0; i+1 < len(parts); i++ {
			if rank, ok := lookup(parts[i], parts[i+1]); ok && (best < 0 || rank < bestRank) {
				best, bestRank = i, rank
			}
		}
		if best < 0 {
			break
		}
		parts[best] += parts[best+1]
		parts = append(parts[:best+1], parts[best+2:]...)
	}
	return parts
}

func TestSharedBPEExactRanks(t *testing.T) {
	for _, test := range []struct {
		name string
		ab   int
		bc   int
		want []string
	}{
		{"adjacent above float32 precision", 1<<24 + 1, 1 << 24, []string{"a", "bc"}},
		{"leftmost equal rank", 1 << 24, 1 << 24, []string{"ab", "c"}},
		{"large integer priority", 1<<30 + 1, 1 << 30, []string{"a", "bc"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ranks := map[[2]string]int{{"a", "b"}: test.ab, {"b", "c"}: test.bc}
			lookup := func(left, right string) (int, bool) {
				rank, ok := ranks[[2]string{left, right}]
				return rank, ok
			}
			got := MergeBPE([]string{"a", "b", "c"}, lookup)
			if !slices.Equal(got, test.want) {
				t.Fatalf("pieces=%q want=%q", got, test.want)
			}
			vocab := &Vocab{mergeRank: map[pair]int{
				{left: "a", right: "b"}: test.ab,
				{left: "b", right: "c"}: test.bc,
			}}
			if got := vocab.applyBPE("abc"); !slices.Equal(got, test.want) {
				t.Fatalf("native pieces=%q want=%q", got, test.want)
			}
		})
	}
	vocab := &Vocab{IgnoreMerges: true, tokenToID: map[string]TokenID{"abc": 7}}
	if got := vocab.applyBPE("abc"); !slices.Equal(got, []string{"abc"}) {
		t.Fatalf("IgnoreMerges pieces=%q", got)
	}
}

func TestSharedBPEMergeReference(t *testing.T) {
	alphabet := []string{"a", "b", "é", "😀", "\xff"}
	rng := rand.New(rand.NewPCG(41, 53))
	ranks := make(map[[2]string]int)
	for _, left := range alphabet {
		for _, right := range alphabet {
			ranks[[2]string{left, right}] = rng.IntN(8)
		}
	}
	for _, left := range alphabet {
		for _, right := range alphabet {
			for _, third := range alphabet {
				ranks[[2]string{left + right, third}] = rng.IntN(8)
			}
		}
	}
	lookup := func(left, right string) (int, bool) {
		rank, ok := ranks[[2]string{left, right}]
		return rank, ok
	}
	for trial := range 300 {
		input := make([]string, trial%65)
		for index := range input {
			input[index] = alphabet[rng.IntN(len(alphabet))]
		}
		original := slices.Clone(input)
		got := MergeBPE(input, lookup)
		want := scanMergeBPE(input, lookup)
		if !slices.Equal(got, want) {
			t.Fatalf("trial=%d input=%q got=%q want=%q", trial, input, got, want)
		}
		if !slices.Equal(input, original) {
			t.Fatal("input symbols changed")
		}
	}
}

func TestSharedBPEWorkBound(t *testing.T) {
	for _, length := range []int{256, 512, 1024, 4096} {
		input := make([]string, length)
		for index := range input {
			input[index] = []string{"a", "b", "c", "d"}[index%4]
		}
		lookups := 0
		pieces := MergeBPE(input, func(left, right string) (int, bool) {
			lookups++
			if len(left) != 1 || len(right) != 1 {
				return 0, false
			}
			return int(left[0]), true
		})
		if len(pieces) != length/2 || lookups > 3*(length-1) {
			t.Fatalf("length=%d pieces=%d lookups=%d exceeds linear candidate bound=%d", length, len(pieces), lookups, 3*(length-1))
		}
	}
}
