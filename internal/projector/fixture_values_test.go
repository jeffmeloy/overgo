package projector

const (
	fixtureSmallPixelBudget  = 16
	fixtureMediumPixelBudget = 32
	fixtureLargePixelBudget  = 64
	fixtureMaxAspectRatio    = 10
	fixtureOpaqueAlpha       = 255
	fixtureChannelModulus    = 256
)

func fixtureVisionBackbone(imageSize, patchSize, hidden, intermediate, layers, heads int) visionBackboneSpec {
	return visionBackboneSpec{
		ImageSize: imageSize, PatchSize: patchSize, Hidden: hidden, Intermediate: intermediate,
		Layers: layers, Heads: heads, LayerNormEpsilon: 1e-6, ImageStd: [3]float32{1, 1, 1},
	}
}
