package diffusionimage

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestTransformerBlockProgramMatchesTorchFixture(t *testing.T) {
	fixture := loadGolden[transformerBlockFixture](t, "transformerblock")
	program := testcheck.Must(compileTransformerBlockProgram(fixture.block(), fixture.model().Cfg, fixture.C, fixture.H*fixture.W))(t)
	got := testcheck.Must(program.execute(t.Context(), nil, fixture.X))(t)
	requireWithin(t, "transformer graph", got, fixture.Y, tolLoose)
}
