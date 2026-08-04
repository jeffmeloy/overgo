package model

const (
	plamo2MinimumTimeStepWidth uint32 = 64
	plamo2TimeStepWidthRatio   uint32 = 16
)

func plamo2TimeStepWidth(embeddingLength uint32) uint64 {
	return uint64(max(plamo2MinimumTimeStepWidth, embeddingLength/plamo2TimeStepWidthRatio))
}
