package sampling

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"testing"

	"overgo/internal/processmeasure"
)

// Experimental only: preserve the legacy FNV-1a stream without rereading its
// vocabulary suffix on every compilation. For h = 256*q+r, a byte XOR changes
// only r. Multiplication by the FNV prime propagates q linearly, and the low byte
// evolves independently. Thus a fixed N-byte suffix maps any incoming h to
// (h &^ 255)*prime^N + table[byte(h)] modulo 2^64. The table needs every byte
// state, not one entry per grammar. Building it still costs 256 hash lanes per
// suffix byte; this study measures that cost before any runtime promotion.
type vocabularyHashSuffixStudy struct {
	factor uint64
	table  [math.MaxUint8 + 1]uint64
}

func newVocabularyHashSuffixStudy(data []byte) vocabularyHashSuffixStudy {
	// FNV-1a's 64-bit prime, identical to hash/fnv's specified algorithm.
	const prime uint64 = 1099511628211
	suffix := vocabularyHashSuffixStudy{factor: 1}
	for i := range suffix.table {
		suffix.table[i] = uint64(i)
	}
	for _, value := range data {
		for i := range suffix.table {
			suffix.table[i] = (suffix.table[i] ^ uint64(value)) * prime
		}
		suffix.factor *= prime
	}
	return suffix
}

func (suffix *vocabularyHashSuffixStudy) apply(prefix uint64) uint64 {
	return (prefix&^math.MaxUint8)*suffix.factor + suffix.table[byte(prefix)]
}

func TestGBNFVocabularyHashCompatibilityStudy(t *testing.T) {
	t.Run("standard library oracle for every incoming byte state", func(t *testing.T) {
		alphabet := make([]byte, math.MaxUint8+1)
		for i := range alphabet {
			alphabet[i] = byte(i)
		}
		for _, suffix := range [][]byte{
			nil, {0}, {math.MaxUint8}, alphabet,
			bytes.Repeat([]byte{0}, len(alphabet)),
			bytes.Repeat([]byte{math.MaxUint8}, len(alphabet)),
			bytes.Repeat(alphabet, len(alphabet)),
		} {
			transform := newVocabularyHashSuffixStudy(suffix)
			// Each prefix followed by every possible byte exercises every
			// incoming low-byte state (the FNV prime is odd), with varied high
			// bits and carries. The oracle retains the ordinary sequential hash.
			for _, prefix := range []string{"", "root ::= \"a\"", "alternate ::= [α-ω]+", string(alphabet)} {
				var seen [math.MaxUint8 + 1]bool
				for _, next := range alphabet {
					reference := fnv.New64a()
					_, _ = reference.Write([]byte(prefix))
					_, _ = reference.Write([]byte{next})
					incoming := reference.Sum64()
					seen[byte(incoming)] = true
					_, _ = reference.Write(suffix)
					if got := transform.apply(incoming); got != reference.Sum64() {
						t.Fatalf("suffix bytes=%d prefix=%q next=%d got=%x want=%x", len(suffix), prefix, next, got, reference.Sum64())
					}
				}
				for state, covered := range seen {
					if !covered {
						t.Fatalf("incoming state %d was not exercised", state)
					}
				}
			}
		}
	})

	t.Run("serialized vocabulary and grammar prefixes", func(t *testing.T) {
		pieces := bytePieces("", "a", "\xc3", "\xa9", "\x00\xff", "</s>")
		data := vocabularyHashStudyBytes(pieces, []bool{false, false, false, false, false, true})
		transform := newVocabularyHashSuffixStudy(data)
		retained := bytes.Clone(data)
		clear(data) // Application owns no reference to the serialized vocabulary.
		for i := range (math.MaxUint8 + 1) * 2 {
			reference := fnv.New64a()
			for _, part := range []string{fmt.Sprintf("root ::= \"%d\"", i), "root"} {
				_, _ = reference.Write(binary.LittleEndian.AppendUint64(nil, uint64(len(part))))
				_, _ = reference.Write([]byte(part))
			}
			incoming := reference.Sum64()
			_, _ = reference.Write(retained)
			if got := transform.apply(incoming); got != reference.Sum64() {
				t.Fatalf("grammar=%d got=%x want=%x", i, got, reference.Sum64())
			}
		}
		if allocations := testing.AllocsPerRun(10, func() { _ = transform.apply(math.MaxUint64) }); allocations != 0 {
			t.Fatalf("suffix application allocations=%g", allocations)
		}
	})

	t.Run("paired hash cost including table construction", func(t *testing.T) {
		// Probe sizes describe synthetic vocabularies, not production model
		// measurements. One compile exposes cold cost; twice the byte-state
		// count exposes amortization across many distinct grammar prefixes.
		// ABBA retains matching inputs and order. No wall-time threshold, total
		// inference claim, or runtime promotion follows from these observations.
		for _, size := range []int{8192, 131072} {
			pieces := make([][]byte, size)
			eos := make([]bool, size)
			for i := range pieces {
				pieces[i] = []byte(fmt.Sprint(i))
			}
			eos[len(eos)-1] = true
			data := vocabularyHashStudyBytes(pieces, eos)
			for _, calls := range []int{1, (math.MaxUint8 + 1) * 2} {
				prefixes := make([][]byte, calls)
				for i := range prefixes {
					prefixes[i] = []byte(fmt.Sprintf("root ::= \"%d\"", i))
				}
				var oracle []uint64
				for sample, transformed := range []bool{false, true, true, false} {
					results := make([]uint64, calls)
					started, err := processmeasure.Counter()
					if err != nil {
						t.Fatal(err)
					}
					var transform vocabularyHashSuffixStudy
					if transformed {
						transform = newVocabularyHashSuffixStudy(data)
					}
					for i, prefix := range prefixes {
						hash := fnv.New64a()
						_, _ = hash.Write(prefix)
						if transformed {
							results[i] = transform.apply(hash.Sum64())
						} else {
							_, _ = hash.Write(data)
							results[i] = hash.Sum64()
						}
					}
					finished, err := processmeasure.Counter()
					if err != nil {
						t.Fatal(err)
					}
					if sample == 0 {
						oracle = results
					}
					for i, want := range oracle {
						if results[i] != want {
							t.Fatalf("cost sample=%d grammar=%d changed signature", sample, i)
						}
					}
					t.Logf("hash-only tokens=%d wire_bytes=%d grammars=%d sample=%d transform=%t includes_setup=true wall=%s", size, len(data), calls, sample, transformed, finished-started)
				}
			}
		}
	})
}

func vocabularyHashStudyBytes(pieces [][]byte, eos []bool) []byte {
	var data []byte
	for i, piece := range pieces {
		data = binary.LittleEndian.AppendUint64(data, uint64(len(piece)))
		data = append(data, piece...)
		terminal := byte(0)
		if eos[i] {
			terminal = 1
		}
		data = append(data, terminal)
	}
	return data
}
