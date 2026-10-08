package seq2seq

import (
	"cmp"
	"container/heap"
	"fmt"
	"os"
	"slices"
	"strings"
	"unicode/utf8"

	"overgo/internal/tokenizer"
)

// A SentencePiece model file says how its pieces are to be used: as a unigram
// table, segmented by the best-scoring cover of the text, or as byte-pair
// merges, joined two symbols at a time in the order training made them. The
// two give different ids for the same text over the same pieces. Needle's
// file declares merges, and its text was once segmented as a unigram table:
// every request then carried ids the model was not trained on, and no
// comparison of the forward pass, which starts after the ids, could see it.
// This file reads what the model file declares and segments as it says.

// The fields of the model file this reader takes, by their numbers in
// SentencePiece's ModelProto, and the wire types they are written in.
const (
	wireVarint, wireFixed64, wireBytes, wireFixed32 = 0, 1, 2, 5
	wireBits                                        = 3
	fixed32Bytes, fixed64Bytes                      = 4, 8
	varintBits, varintLimit                         = 7, 64
	varintMore, varintValue                         = 0x80, 0x7f

	modelPieces, modelTrainer, modelNormalizer = 1, 2, 3
	pieceText, pieceType                       = 1, 3
	trainerModelType                           = 3
	normalizerCollapseSpaces                   = 4

	// modelTypeMerges is the model type of byte-pair merges.
	modelTypeMerges = 2
	// pieceUserDefined is a piece that text holds whole and that joins nothing;
	// pieceNormal is one segmentation may produce; pieceByte stands for one byte
	// of a character no piece spells. A control, an unknown or a byte piece is
	// never read out of text that happens to spell its name.
	pieceNormal, pieceUserDefined, pieceByte = 1, 4, 6
)

// spaceMarker stands for a space in a piece, and opens the text.
const spaceMarker = "▁"

// protoFields walks one protobuf message, giving each field's number with its
// integer value or its bytes.
func protoFields(message []byte, visit func(field int, value uint64, body []byte) error) error {
	for at := 0; at < len(message); {
		key, next, err := protoVarint(message, at)
		if err != nil {
			return err
		}
		at = next
		field, wire := int(key>>wireBits), int(key&(1<<wireBits-1))
		var value uint64
		var body []byte
		switch wire {
		case wireVarint:
			if value, at, err = protoVarint(message, at); err != nil {
				return err
			}
		case wireBytes:
			size, start, err := protoVarint(message, at)
			if err != nil || size > uint64(len(message)-start) {
				return fmt.Errorf("seq2seq: tokenizer model field %d is truncated", field)
			}
			body, at = message[start:start+int(size)], start+int(size)
		case wireFixed32, wireFixed64:
			width := fixed32Bytes
			if wire == wireFixed64 {
				width = fixed64Bytes
			}
			if at+width > len(message) {
				return fmt.Errorf("seq2seq: tokenizer model field %d is truncated", field)
			}
			at += width
		default:
			return fmt.Errorf("seq2seq: tokenizer model field %d has wire type %d", field, wire)
		}
		if err := visit(field, value, body); err != nil {
			return err
		}
	}
	return nil
}

func protoVarint(message []byte, at int) (uint64, int, error) {
	var value uint64
	for shift := 0; shift < varintLimit; shift += varintBits {
		if at >= len(message) {
			break
		}
		octet := message[at]
		at++
		value |= uint64(octet&varintValue) << shift
		if octet < varintMore {
			return value, at, nil
		}
	}
	return 0, 0, fmt.Errorf("seq2seq: tokenizer model holds a truncated integer")
}

// pieceMerges segments text the way a SentencePiece model of byte-pair merges
// does: the text is cut into characters, a user-defined symbol kept whole, and
// the two neighbours whose joined piece training made earliest are joined,
// again and again, until no two neighbours spell a piece.
type pieceMerges struct {
	ids      map[string]int
	bytes    map[byte]int
	scores   []float64
	symbols  []string
	unknown  int
	collapse bool
}

// loadPieceMerges reads the model file at path. merges is false for a model
// that does not declare byte-pair merges, which its caller segments as a
// unigram table.
func loadPieceMerges(path string, pieces []tokenizer.UnigramPiece, unknown int) (table *pieceMerges, merges bool, err error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, false, err
	}
	// A model file leaves out what it holds at its default, and collapsing runs
	// of spaces is the default: only a file that says otherwise turns it off.
	table = &pieceMerges{ids: make(map[string]int, len(pieces)), bytes: map[byte]int{}, scores: make([]float64, len(pieces)), unknown: unknown, collapse: true}
	index := 0
	err = protoFields(raw, func(field int, _ uint64, body []byte) error {
		switch field {
		case modelPieces:
			if index >= len(pieces) {
				return fmt.Errorf("seq2seq: tokenizer model holds more pieces than were read")
			}
			text, kind := "", uint64(pieceNormal)
			if err := protoFields(body, func(field int, value uint64, body []byte) error {
				switch field {
				case pieceText:
					text = string(body)
				case pieceType:
					kind = value
				}
				return nil
			}); err != nil {
				return err
			}
			if text != pieces[index].Piece {
				return fmt.Errorf("seq2seq: tokenizer model piece %d reads %q and %q", index, text, pieces[index].Piece)
			}
			table.scores[index] = pieces[index].Score
			if kind == pieceNormal || kind == pieceUserDefined {
				table.ids[text] = index
			}
			if kind == pieceUserDefined {
				table.symbols = append(table.symbols, text)
			}
			var octet byte
			if _, err := fmt.Sscanf(text, "<0x%02X>", &octet); kind == pieceByte && err == nil {
				table.bytes[octet] = index
			}
			index++
		case modelTrainer:
			return protoFields(body, func(field int, value uint64, _ []byte) error {
				merges = merges || field == trainerModelType && value == modelTypeMerges
				return nil
			})
		case modelNormalizer:
			return protoFields(body, func(field int, value uint64, _ []byte) error {
				if field == normalizerCollapseSpaces {
					table.collapse = value != 0
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, false, err
	}
	if index != len(pieces) {
		return nil, false, fmt.Errorf("seq2seq: tokenizer model holds %d pieces of %d", index, len(pieces))
	}
	// A longer symbol is tried before one it begins with.
	slices.SortFunc(table.symbols, func(left, right string) int { return cmp.Compare(len(right), len(left)) })
	return table, merges, nil
}

// joinable is one pair of neighbours that spell a piece.
type joinable struct {
	left, right int
	score       float64
	size        int
}

// joinables orders pairs by the score of the piece they spell, the higher
// first, and the leftmost of two alike first, as SentencePiece does.
type joinables []joinable

// Len is how many pairs wait.
func (queue joinables) Len() int { return len(queue) }

// Less puts the pair that is joined sooner first.
func (queue joinables) Less(left, right int) bool {
	if queue[left].score != queue[right].score {
		return queue[left].score > queue[right].score
	}
	return queue[left].left < queue[right].left
}

// Swap exchanges two waiting pairs.
func (queue joinables) Swap(left, right int) { queue[left], queue[right] = queue[right], queue[left] }

// Push adds a pair.
func (queue *joinables) Push(value any) { *queue = append(*queue, value.(joinable)) }

// Pop takes the last pair, which the heap has put there to be taken.
func (queue *joinables) Pop() any {
	last := len(*queue) - 1
	value := (*queue)[last]
	*queue = (*queue)[:last]
	return value
}

// encode gives text's ids. Runs of spaces are collapsed and the ends trimmed
// where the model says so, the text is opened with the space marker, and each
// space becomes one. A character no piece spells is written as its bytes.
func (table *pieceMerges) encode(text string) ([]int, error) {
	if table.collapse {
		text = strings.Join(strings.FieldsFunc(text, func(char rune) bool { return char == ' ' }), " ")
	}
	if text == "" {
		return nil, nil
	}
	text = spaceMarker + strings.ReplaceAll(text, " ", spaceMarker)
	type symbol struct {
		text           string
		previous, next int
		whole          bool
	}
	var symbols []symbol
	for offset := 0; offset < len(text); {
		size, whole := 0, false
		for _, defined := range table.symbols {
			if strings.HasPrefix(text[offset:], defined) {
				size, whole = len(defined), true
				break
			}
		}
		if !whole {
			_, size = utf8.DecodeRuneInString(text[offset:])
		}
		symbols = append(symbols, symbol{text: text[offset : offset+size], previous: len(symbols) - 1, next: len(symbols) + 1, whole: whole})
		offset += size
	}
	symbols[len(symbols)-1].next = -1
	queue := &joinables{}
	offer := func(left, right int) {
		if left < 0 || right < 0 || symbols[left].whole || symbols[right].whole {
			return
		}
		joined := symbols[left].text + symbols[right].text
		if id, known := table.ids[joined]; known {
			heap.Push(queue, joinable{left: left, right: right, score: table.scores[id], size: len(joined)})
		}
	}
	for index := 1; index < len(symbols); index++ {
		offer(index-1, index)
	}
	for queue.Len() > 0 {
		pair := heap.Pop(queue).(joinable)
		left, right := &symbols[pair.left], &symbols[pair.right]
		// A pair offered before one of its sides was joined elsewhere is stale.
		if left.text == "" || right.text == "" || len(left.text)+len(right.text) != pair.size || left.next != pair.right {
			continue
		}
		left.text += right.text
		right.text = ""
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].previous = pair.left
		}
		offer(left.previous, pair.left)
		offer(pair.left, left.next)
	}
	var ids []int
	for index := 0; index >= 0; index = symbols[index].next {
		if id, known := table.ids[symbols[index].text]; known {
			ids = append(ids, id)
			continue
		}
		for _, octet := range []byte(symbols[index].text) {
			id, known := table.bytes[octet]
			if !known {
				id = table.unknown
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}
