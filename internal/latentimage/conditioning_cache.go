package latentimage

import "slices"

// conditioningEntry is one prompt's text encoder output.
type conditioningEntry struct {
	prompt string
	hidden []float32
}

// conditioningCache keeps a session's most recent prompts' text encoder outputs
// on the host, so a prompt met again places no text encoder on the device: a
// negative prompt is usually the same from request to request, and a request
// that changes only its seed changes neither prompt. A session holds one
// checkpoint, tokenizer, template and storage type, so a prompt names the whole
// input; the cache ends with the session. limit bounds the values held: the
// least recently used prompts leave first.
type conditioningCache struct {
	limit, held int
	entries     []conditioningEntry
}

// lookup returns a prompt's output, shared with the cache, and makes it the
// most recently used.
func (c *conditioningCache) lookup(prompt string) ([]float32, bool) {
	at := slices.IndexFunc(c.entries, func(entry conditioningEntry) bool { return entry.prompt == prompt })
	if at < 0 {
		return nil, false
	}
	entry := c.entries[at]
	c.entries = append(slices.Delete(c.entries, at, at+1), entry)
	return entry.hidden, true
}

// store keeps a prompt's output, which the caller no longer writes, and evicts
// what no longer fits under the limit. An output over the limit is not kept.
func (c *conditioningCache) store(prompt string, hidden []float32) {
	if _, known := c.lookup(prompt); known || len(hidden) > c.limit {
		return
	}
	c.entries = append(c.entries, conditioningEntry{prompt: prompt, hidden: hidden})
	c.held += len(hidden)
	for c.held > c.limit {
		c.held -= len(c.entries[0].hidden)
		c.entries = slices.Delete(c.entries, 0, 1)
	}
}
