package model

import "overgo/internal/tensor"

const maxMoETopK = tensor.MaxMoETopK

func exceedsMoETopK(topK uint32) bool {
	return topK > maxMoETopK
}

func validMoESelection(topK, experts uint32) bool {
	return topK > 0 && topK <= experts && !exceedsMoETopK(topK)
}

func validExpertDimensions(spec Spec) bool {
	return validMoESelection(spec.ExpertUsedCount, spec.ExpertCount) && spec.ExpertFeedForward > 0
}

func validExpertRouting(spec Spec) bool {
	return spec.ExpertGatingFunc == expertGatingSoftmax || spec.ExpertGatingFunc == expertGatingSigmoid
}

func validFullRotaryHead(spec Spec) bool {
	return spec.RopeDimensionCount == spec.KeyLength && spec.KeyLength == spec.ValueLength &&
		spec.RopeDimensionCount%rotaryPairAlignment == 0
}
