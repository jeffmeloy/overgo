package projector

const (
	fixtureSmallPixelBudget      = 16
	fixtureMediumPixelBudget     = 32
	fixtureLargePixelBudget      = 64
	fixtureOpaqueAlpha           = 255
	fixtureChannelModulus        = 256
	fixtureSpatialMerge          = 2
	fixtureRopeFrequency         = float32(10000)
	fixtureExtendedRopeFrequency = float32(1_000_000)
	fixtureNormEpsilon           = float32(1e-6)
	fixtureAudioSampleRate       = 16000
)

func fixtureVisionBackbone(imageSize, patchSize, hidden, intermediate, layers, heads int) visionBackboneSpec {
	return visionBackboneSpec{
		ImageSize: imageSize, PatchSize: patchSize, Hidden: hidden, Intermediate: intermediate,
		Layers: layers, Heads: heads, LayerNormEpsilon: fixtureNormEpsilon, RopeFrequency: fixtureRopeFrequency,
		ImageStd: [3]float32{1, 1, 1},
	}
}
