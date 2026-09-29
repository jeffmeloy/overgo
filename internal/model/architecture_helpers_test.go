package model

import "overgo/internal/gguf"

func testProfile(architecture string) ArchitectureProfile {
	profile, _ := LookupArchitecture(architecture)
	return profile
}

func bindFixtureSpec(spec Spec) Spec {
	profile, ok := LookupArchitecture(spec.Architecture)
	if !ok {
		return spec
	}
	return spec.withProfile(profile)
}

func (s Spec) PlanLayer(layer uint32, recurrent bool) LayerPlan {
	plan, err := s.planLayer(s.Profile(), layer, recurrent)
	if err != nil {
		panic(err)
	}
	return plan
}

func readFixtureWeights(file *gguf.File, spec Spec) (Weights, error) {
	return ReadWeights(file, bindFixtureSpec(spec))
}

func hasPostNorm(architecture string) bool {
	return testProfile(architecture).Has(architecturePostNorm)
}

func usesPostOnlyNorm(architecture string) bool {
	return testProfile(architecture).Has(architecturePostOnlyNorm)
}

func usesNormalRoPE(architecture string) bool {
	return testProfile(architecture).Position == PositionNormal
}

func usesParallelResidual(architecture string) bool {
	return testProfile(architecture).Residual == residualParallel
}

func usesFusedGateUp(architecture string) bool {
	return testProfile(architecture).FeedForward == feedForwardFusedGateUp
}

func usesGELU(architecture string) bool {
	profile := testProfile(architecture)
	return profile.FeedForward == feedForwardGELU || profile.FeedForward == feedForwardSequentialGELU
}

func supportsLongRoPE(architecture string) bool {
	return testProfile(architecture).Has(architectureLongRoPE)
}
