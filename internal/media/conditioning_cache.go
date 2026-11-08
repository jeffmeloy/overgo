package media

import "slices"

// conditioningEntry is one text's encoder output, under the encoder that made it.
type conditioningEntry struct {
	encoder, text string
	values        []float32
}

// ConditioningCache keeps text encoders' outputs on the host by what made them, the encoder's identity and the text,
// so a text met again runs no encoder: a request that changes only its seed changes neither of its prompts, and a
// negative prompt is the same from request to request. One cache serves image and video. Limit bounds the values held,
// the least recently used leaving first, and an output over it is not kept; a Kept text, the empty or a model's
// standard negative prompt, never leaves and counts against no limit.
type ConditioningCache struct {
	Limit   int
	Kept    []string
	held    int
	entries []conditioningEntry
}

// Lookup returns a text's output under an encoder, shared with the cache, and makes it the most recently used.
func (c *ConditioningCache) Lookup(encoder, text string) ([]float32, bool) {
	at := slices.IndexFunc(c.entries, func(entry conditioningEntry) bool { return entry.encoder == encoder && entry.text == text })
	if at < 0 {
		return nil, false
	}
	entry := c.entries[at]
	c.entries = append(slices.Delete(c.entries, at, at+1), entry)
	return entry.values, true
}

// Store keeps a text's output under an encoder, which the caller no longer writes, and evicts what no longer fits
// under the limit, kept texts aside.
func (c *ConditioningCache) Store(encoder, text string, values []float32) {
	kept := slices.Contains(c.Kept, text)
	if _, known := c.Lookup(encoder, text); known || !kept && len(values) > c.Limit {
		return
	}
	c.entries = append(c.entries, conditioningEntry{encoder: encoder, text: text, values: values})
	if kept {
		return
	}
	c.held += len(values)
	for c.held > c.Limit {
		at := slices.IndexFunc(c.entries, func(entry conditioningEntry) bool { return !slices.Contains(c.Kept, entry.text) })
		c.held -= len(c.entries[at].values)
		c.entries = slices.Delete(c.entries, at, at+1)
	}
}
