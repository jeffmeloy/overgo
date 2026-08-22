package inference

import (
	"math/bits"

	"overgo/internal/checked"
	"overgo/internal/recipe"
)

func cachePageTokens(tokens uint32, session recipe.SessionPolicy) uint32 {
	var unavailable uint32
	if !checked.Nonzero(tokens) {
		return unavailable
	}
	if session == recipe.SessionRequest {
		return tokens
	}
	reversed := bits.Reverse32(tokens)
	return bits.Reverse32(reversed & -reversed)
}

func cachePageCapacity(tokens, limit uint32, session recipe.SessionPolicy) uint32 {
	var unavailable uint32
	pageTokens := cachePageTokens(tokens, session)
	if !checked.Nonzero(pageTokens) {
		return unavailable
	}
	capacity, valid := checked.RoundUpMultiple(uint64(tokens), uint64(pageTokens))
	if !valid {
		return unavailable
	}
	if capacity > uint64(limit) {
		capacity = uint64(limit)
	}
	return uint32(capacity)
}
