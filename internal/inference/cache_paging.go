package inference

import (
	"math/bits"

	"overgo/internal/recipe"
)

func cachePageTokens(tokens uint32, session recipe.SessionPolicy) uint32 {
	if tokens == 0 {
		return 0
	}
	if session == recipe.SessionRequest {
		return tokens
	}
	reversed := bits.Reverse32(tokens)
	return bits.Reverse32(reversed & -reversed)
}

func cachePageCapacity(tokens, limit uint32, session recipe.SessionPolicy) uint32 {
	pageTokens := cachePageTokens(tokens, session)
	if pageTokens == 0 {
		return 0
	}
	capacity := (uint64(tokens) + uint64(pageTokens) - 1) / uint64(pageTokens) * uint64(pageTokens)
	if capacity > uint64(limit) {
		capacity = uint64(limit)
	}
	return uint32(capacity)
}
