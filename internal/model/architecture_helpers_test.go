package model

func testProfile(architecture string) ArchitectureProfile {
	profile, _ := LookupArchitecture(architecture)
	return profile
}

func hasPostNorm(architecture string) bool {
	return testProfile(architecture).Has(ArchitecturePostNorm)
}

func usesPostOnlyNorm(architecture string) bool {
	return testProfile(architecture).Has(ArchitecturePostOnlyNorm)
}

func usesNormalRoPE(architecture string) bool {
	return testProfile(architecture).Position == PositionNormal
}

func usesParallelResidual(architecture string) bool {
	return testProfile(architecture).Residual == ResidualParallel
}

func usesFusedGateUp(architecture string) bool {
	return testProfile(architecture).FeedForward == FeedForwardFusedGateUp
}

func supportsLongRoPE(architecture string) bool {
	return testProfile(architecture).Has(ArchitectureLongRoPE)
}
