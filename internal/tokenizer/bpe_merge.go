package tokenizer

import "container/heap"

type rankedBigram struct {
	left, right int
	rank        int
	size        int
}

// MergeBPE applies exact integer-ranked adjacent merges. Equal ranks choose
// the leftmost pair. Callers retain ownership of symbol splitting and rank
// lookup, including their different malformed-byte boundaries. Input symbols
// are not modified.
func MergeBPE(input []string, lookup func(left, right string) (rank int, ok bool)) []string {
	symbols := make([]spmSymbol, len(input))
	for index, part := range input {
		symbols[index] = spmSymbol{previous: index - 1, next: index + 1, text: part}
	}
	return mergeBPE(symbols, lookup)
}

// mergeBPE accepts already split symbols so the native GGUF path can build
// them directly without an intermediate string slice.
func mergeBPE(symbols []spmSymbol, lookup func(left, right string) (rank int, ok bool)) []string {
	if len(symbols) == 0 {
		return nil
	}
	symbols[len(symbols)-1].next = -1
	queue := mergeQueue[rankedBigram]{
		values: make([]rankedBigram, 0, len(symbols)),
		less: func(left, right rankedBigram) bool {
			if left.rank == right.rank {
				return left.left < right.left
			}
			return left.rank < right.rank
		},
	}
	push := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		rank, ok := lookup(symbols[left].text, symbols[right].text)
		if !ok {
			return
		}
		heap.Push(&queue, rankedBigram{
			left: left, right: right, rank: rank,
			size: len(symbols[left].text) + len(symbols[right].text),
		})
	}
	for index := 1; index < len(symbols); index++ {
		push(index-1, index)
	}
	for queue.Len() > 0 {
		bigram := heap.Pop(&queue).(rankedBigram)
		left, right := &symbols[bigram.left], &symbols[bigram.right]
		if left.text == "" || right.text == "" || len(left.text)+len(right.text) != bigram.size {
			continue
		}
		left.text += right.text
		right.text = ""
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].previous = bigram.left
		}
		push(left.previous, bigram.left)
		push(bigram.left, left.next)
	}
	result := make([]string, 0, len(symbols))
	for index := 0; index >= 0; index = symbols[index].next {
		result = append(result, symbols[index].text)
	}
	return result
}
