package inference

import "overgo/internal/tokenizer"

// commonTokenPrefix counts the leading tokens two encodings share.
func commonTokenPrefix(left, right []tokenizer.TokenID) int {
	count := 0
	for count < len(left) && count < len(right) && left[count] == right[count] {
		count++
	}
	return count
}
