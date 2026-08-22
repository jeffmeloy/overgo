package tokenizer

// InfillPrefixShare applies the canonical three-to-one prefix/suffix budget.
func InfillPrefixShare(total int) int { return 3 * total / 4 }

// LastInvocation locates the final invocation token sequence in token IDs.
func LastInvocation(ids []TokenID, invocation []uint32) (int, bool) {
	if len(invocation) == 0 || len(ids) < len(invocation) {
		return 0, false
	}
	for remaining := len(ids) - len(invocation) + 1; remaining > 0; remaining-- {
		start := remaining - 1
		matched := true
		for offset, expected := range invocation {
			id := ids[start+offset]
			if id == NullToken || uint32(id) != expected {
				matched = false
				break
			}
		}
		if matched {
			return start, true
		}
	}
	return 0, false
}
