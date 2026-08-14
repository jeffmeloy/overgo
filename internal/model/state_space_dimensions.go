package model

const (
	minimumReducedTimeStepWidth uint32 = 64
	reducedTimeStepWidthRatio   uint32 = 16
)

func reducedTimeStepWidth(embeddingLength uint32) uint64 {
	return uint64(max(minimumReducedTimeStepWidth, embeddingLength/reducedTimeStepWidthRatio))
}
