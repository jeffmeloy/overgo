package inference

// DefaultCachePageTokens: retained KV-cache page width.
const DefaultCachePageTokens uint32 = 256

func resolveCachePageTokens(pageTokens uint32) uint32 {
	if pageTokens == 0 {
		return DefaultCachePageTokens
	}
	return pageTokens
}
