package tokenizer

import (
	"container/heap"
	"fmt"
	"unicode/utf8"
)

type spmSymbol struct {
	previous int
	next     int
	text     string
}

type spmBigram struct {
	left  int
	right int
	score float32
	size  int
}

func (v *Vocab) encodeSPM(text string) ([]TokenID, error) {
	if text == "" {
		return nil, nil
	}
	symbols := make([]spmSymbol, 0, utf8.RuneCountInString(text))
	for offset := 0; offset < len(text); {
		_, size := utf8.DecodeRuneInString(text[offset:])
		if size == 0 {
			break
		}
		index := len(symbols)
		symbols = append(symbols, spmSymbol{
			previous: index - 1,
			next:     index + 1,
			text:     text[offset : offset+size],
		})
		offset += size
	}
	symbols[len(symbols)-1].next = -1
	queue := mergeQueue[spmBigram]{
		values: make([]spmBigram, 0, len(symbols)),
		less: func(left, right spmBigram) bool {
			if left.score == right.score {
				return left.left < right.left
			}
			return left.score > right.score
		},
	}
	reverseMerge := make(map[string][2]int)
	addBigram := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		combined := symbols[left].text + symbols[right].text
		id, ok := v.tokenToID[combined]
		if !ok || id < 0 || int(id) >= len(v.Tokens) {
			return
		}
		heap.Push(&queue, spmBigram{
			left:  left,
			right: right,
			score: v.Tokens[id].Score,
			size:  len(combined),
		})
		reverseMerge[combined] = [2]int{left, right}
	}
	for index := 1; index < len(symbols); index++ {
		addBigram(index-1, index)
	}
	for queue.Len() > 0 {
		bigram := heap.Pop(&queue).(spmBigram)
		left := &symbols[bigram.left]
		right := &symbols[bigram.right]
		if left.text == "" || right.text == "" ||
			len(left.text)+len(right.text) != bigram.size {
			continue
		}
		left.text += right.text
		right.text = ""
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].previous = bigram.left
		}
		addBigram(left.previous, bigram.left)
		addBigram(bigram.left, left.next)
	}
	output := make([]TokenID, 0, len(symbols))
	var emit func(int) error
	emit = func(index int) error {
		symbol := symbols[index]
		if id, ok := v.tokenToID[symbol.text]; ok {
			output = append(output, id)
			return nil
		}
		if children, ok := reverseMerge[symbol.text]; ok {
			if err := emit(children[0]); err != nil {
				return err
			}
			return emit(children[1])
		}
		for _, value := range []byte(symbol.text) {
			text := fmt.Sprintf("<0x%02X>", value)
			id, ok := v.tokenToID[text]
			if !ok {
				id, ok = v.tokenToID[string([]byte{value})]
			}
			if !ok {
				return fmt.Errorf("tokenizer: SPM has no token for byte 0x%02X", value)
			}
			output = append(output, id)
		}
		return nil
	}
	for index := 0; index >= 0; index = symbols[index].next {
		if err := emit(index); err != nil {
			return nil, err
		}
	}
	return output, nil
}
