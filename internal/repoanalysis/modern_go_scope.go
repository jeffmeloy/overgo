package repoanalysis

import (
	"slices"
	"strings"
)

var modernGoNumericRuntimePrefixes = []string{
	"cmd/dit-train-probe/", "cmd/flow-organ-train-probe/", "cmd/mot-train-probe/",
	"cmd/oscillatorimage-train-probe/", "cmd/sensenovaparity/", "cmd/seriesforecast-train-probe/",
	"cmd/vqaparity/",
	"internal/adaptiveparity/", "internal/composition/", "internal/cuda/", "internal/densecausal/",
	"internal/devicemath/", "internal/diffusionimage/", "internal/hostmath/", "internal/hybridtrain/",
	"internal/inference/", "internal/latentimage/", "internal/latentvideo/", "internal/media/",
	"internal/model/", "internal/optimizer/", "internal/organ/", "internal/oscillatorimage/",
	"internal/patchtower/", "internal/projector/", "internal/quant/", "internal/representation/",
	"internal/routedlm/", "internal/sampling/", "internal/scratchmodel/", "internal/seq2seq/",
	"internal/seriesforecast/", "internal/speechsynth/", "internal/tabularicl/", "internal/tensor/",
	"internal/tensorstats/", "internal/thoughtbank/",
}

func modernGoNumericRuntimePath(name string) bool {
	return slices.ContainsFunc(modernGoNumericRuntimePrefixes, func(prefix string) bool {
		return strings.HasPrefix(name, prefix)
	})
}
