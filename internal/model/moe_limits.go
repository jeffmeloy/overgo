package model

import "overgo/internal/tensor"

func validMoESelection(topK, experts uint32) bool {
	return topK > tensor.FirstOffset && topK <= experts && topK <= tensor.MaxMoETopK
}

func validExpertDimensions(spec Spec) bool {
	return validMoESelection(spec.ExpertUsedCount, spec.ExpertCount) && spec.ExpertFeedForward > tensor.FirstOffset
}

func validExpertRouting(spec Spec) bool {
	return spec.ExpertGatingFunc == expertGatingSoftmax || spec.ExpertGatingFunc == expertGatingSigmoid
}

func validFullRotaryHead(spec Spec) bool {
	return spec.RopeDimensionCount == spec.KeyLength && spec.KeyLength == spec.ValueLength &&
		spec.RopeDimensionCount%rotaryPairAlignment == tensor.FirstOffset
}
