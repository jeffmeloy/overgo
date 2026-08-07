package inference

// DefaultCachePageTokens: retained KV-cache page width.
const DefaultCachePageTokens uint32 = 256

func resolveCachePageTokens(pageTokens uint32) uint32 {
	if pageTokens == 0 {
		return DefaultCachePageTokens
	}
	return pageTokens
}

func cachePageCapacity(tokens, pageTokens, limit uint32) uint32 {
	pageTokens = resolveCachePageTokens(pageTokens)
	capacity := (uint64(tokens) + uint64(pageTokens) - 1) / uint64(pageTokens) * uint64(pageTokens)
	if capacity > uint64(limit) {
		capacity = uint64(limit)
	}
	return uint32(capacity)
}
