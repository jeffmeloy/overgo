package model

import "llamacpp2go/internal/tensor"

const maxMoETopK = tensor.MaxMoETopK

func exceedsMoETopK(topK uint32) bool {
	return topK > maxMoETopK
}

func validMoESelection(topK, experts uint32) bool {
	return topK > 0 && topK <= experts && !exceedsMoETopK(topK)
}
