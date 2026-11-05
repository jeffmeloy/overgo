package diffusionimage

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestTransformerBlockProgramMatchesTorchFixture(t *testing.T) {
	fixture := loadGolden[transformerBlockFixture](t, "transformerblock")
	program, err := compileTransformerBlockProgram(fixture.block(), fixture.model().Cfg, fixture.C, fixture.H*fixture.W)
	testcheck.NoError(t, err)
	got, err := program.execute(t.Context(), nil, fixture.X)
	testcheck.NoError(t, err)
	requireWithin(t, "transformer graph", got, fixture.Y, tolLoose)
}
