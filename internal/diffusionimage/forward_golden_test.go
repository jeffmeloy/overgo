package diffusionimage

import (
	"math"
	"testing"

	"overgo/internal/testutil"
)

func TestXATGLUGateMatchesTorch(t *testing.T) {
	fx := loadGolden[struct {
		Rows      int       `json:"rows"`
		OutDim    int       `json:"out_dim"`
		Alpha     float64   `json:"alpha"`
		Projected []float32 `json:"projected"`
		Out       []float32 `json:"out"`
	}](t, "xatglu")
	got := make([]float32, fx.Rows*fx.OutDim)
	xatgluGateInto(got, fx.Projected, fx.Alpha, fx.Rows, fx.OutDim)
	requireWithin(t, "xATGLU", got, fx.Out, tolTight)
}

func TestCPLinearMatchesTorch(t *testing.T) {
	fx := loadGolden[struct {
		Tokens int       `json:"bt"`
		NHead  int       `json:"n_head"`
		HeadD  int       `json:"head_dim"`
		QRank  int       `json:"q_rank"`
		Rank   int       `json:"rank"`
		AQ     []float32 `json:"A_q"`
		AK     []float32 `json:"A_k"`
		AV     []float32 `json:"A_v"`
		BQ     []float32 `json:"B_q"`
		BK     []float32 `json:"B_k"`
		BV     []float32 `json:"B_v"`
		Q      []float32 `json:"q"`
		K      []float32 `json:"k"`
		V      []float32 `json:"v"`
	}](t, "cplinear")
	q := make([]float32, fx.Tokens*fx.NHead*fx.HeadD)
	k := make([]float32, fx.Tokens*fx.NHead*fx.HeadD)
	v := make([]float32, fx.Tokens*fx.NHead*fx.HeadD)
	cpFactorContractInto(q, fx.AQ, fx.BQ, fx.Tokens, fx.NHead, fx.QRank, fx.HeadD)
	cpFactorContractInto(k, fx.AK, fx.BK, fx.Tokens, fx.NHead, fx.Rank, fx.HeadD)
	cpFactorContractInto(v, fx.AV, fx.BV, fx.Tokens, fx.NHead, fx.Rank, fx.HeadD)
	requireWithin(t, "CPLinear q", q, fx.Q, tolLoose)
	requireWithin(t, "CPLinear k", k, fx.K, tolLoose)
	requireWithin(t, "CPLinear v", v, fx.V, tolLoose)
}

// TestRoPEMatchesTorch: the golden ships torch cos/sin ([T, hd/2]); the port
// builds vendor-convention tables internally, so the fixture tables are
// expanded to half-split with NEGATED sin (the vendor rotation direction).
func TestRoPEMatchesTorch(t *testing.T) {
	fx := loadGolden[struct {
		T     int       `json:"T"`
		NHead int       `json:"nhead"`
		HD    int       `json:"hd"`
		Cos   []float32 `json:"cos"`
		Sin   []float32 `json:"sin"`
		X     []float32 `json:"x"`
		Y     []float32 `json:"y"`
	}](t, "rope")
	half := fx.HD / 2
	cos := make([]float32, fx.T*fx.HD)
	sin := make([]float32, fx.T*fx.HD)
	for p := 0; p < fx.T; p++ {
		for i := 0; i < half; i++ {
			c, s := fx.Cos[p*half+i], fx.Sin[p*half+i]
			cos[p*fx.HD+i], cos[p*fx.HD+i+half] = c, c
			sin[p*fx.HD+i], sin[p*fx.HD+i+half] = -s, -s
		}
	}
	got := append([]float32(nil), fx.X...)
	applyRope(got, cos, sin, fx.T, fx.NHead, fx.HD)
	requireWithin(t, "RoPE", got, fx.Y, tolTight)

	// The built tables must agree with the golden's (negated) tables.
	builtCos, builtSin := buildRopeTables(fx.T, fx.HD, vendorRopeTheta)
	requireWithin(t, "RoPE cos table", builtCos, cos, tolTight)
	requireWithin(t, "RoPE sin table", builtSin, sin, tolTight)
}

type tpaFixture struct {
	T      int       `json:"T"`
	NEmbd  int       `json:"n_embd"`
	NHead  int       `json:"n_head"`
	HeadD  int       `json:"head_dim"`
	QRank  int       `json:"q_rank"`
	KVRank int       `json:"kv_rank"`
	WAq    []float32 `json:"W_A_q"`
	WAk    []float32 `json:"W_A_k"`
	WAv    []float32 `json:"W_A_v"`
	WBq    []float32 `json:"W_B_q"`
	WBk    []float32 `json:"W_B_k"`
	WBv    []float32 `json:"W_B_v"`
	Woproj []float32 `json:"W_oproj"`
	Boproj []float32 `json:"b_oproj"`
	AlphaO float64   `json:"alpha_o"`
	X      []float32 `json:"x"`
	Y      []float32 `json:"y"`
}

func (fx tpaFixture) weights() tpaWeights {
	return tpaWeights{
		WAq: fx.WAq, WAk: fx.WAk, WAv: fx.WAv,
		WBq: fx.WBq, WBk: fx.WBk, WBv: fx.WBv,
		Woproj: fx.Woproj, Boproj: fx.Boproj, AlphaO: fx.AlphaO,
	}
}

func TestTPABlockMatchesTorch(t *testing.T) {
	fx := loadGolden[tpaFixture](t, "tpa")
	got := make([]float32, fx.T*fx.NEmbd)
	tpaForward(got, fx.X, fx.weights(), fx.T, fx.NEmbd, fx.NHead, fx.HeadD, fx.QRank, fx.KVRank, vendorRopeTheta)
	requireWithin(t, "TPA block", got, fx.Y, tolLoose)
}

func TestGroupNormMatchesTorch(t *testing.T) {
	fx := loadGolden[struct {
		N      int       `json:"N"`
		C      int       `json:"C"`
		H      int       `json:"H"`
		W      int       `json:"W"`
		Groups int       `json:"groups"`
		Eps    float64   `json:"eps"`
		Weight []float32 `json:"weight"`
		Bias   []float32 `json:"bias"`
		X      []float32 `json:"x"`
		Y      []float32 `json:"y"`
	}](t, "groupnorm")
	got := make([]float32, len(fx.X))
	groupNormInto(got, fx.X, fx.Weight, fx.Bias, fx.N, fx.C, fx.H, fx.W, fx.Groups, fx.Eps)
	requireWithin(t, "GroupNorm", got, fx.Y, tolLoose)
}

type resBlockFixture struct {
	N, C, H, W, Groups int
	Eps, ResidScale    float64
	N1W, N1B, C1W, C1B []float32
	N2W, N2B, C2W, C2B []float32
	X, Y               []float32
}

func (fx resBlockFixture) block() *resBlock {
	return &resBlock{
		name:        "block",
		norm1Weight: fx.N1W, norm1Bias: fx.N1B, conv1Weight: fx.C1W, conv1Bias: fx.C1B,
		norm2Weight: fx.N2W, norm2Bias: fx.N2B, conv2Weight: fx.C2W, conv2Bias: fx.C2B,
		residualScale: float32(fx.ResidScale),
	}
}

func (fx resBlockFixture) model() *Model {
	return &Model{Cfg: Config{Groups: fx.Groups, NormEps: fx.Eps}}
}

func TestResBlockMatchesTorch(t *testing.T) {
	fx := loadGolden[resBlockFixture](t, "resblock")
	got := fx.block().forward(fx.model(), fx.X, fx.N, fx.C, fx.H, fx.W)
	requireWithin(t, "ResBlock", got, fx.Y, tolLoose)
}

type transformerBlockFixture struct {
	N, C, H, W, Heads, HeadDim, QRank, KVRank int
	Eps, AttnScale, MLPScale                  float64
	Norm1W, Norm1B, Norm2W, Norm2B            []float32
	WAq, WAk, WAv, WBq, WBk, WBv              []float32
	Woproj, Boproj                            []float32
	AlphaO                                    float64
	MLPGateW, MLPDownW                        []float32
	MLPAlpha                                  float64
	X, Y                                      []float32
}

func (fx transformerBlockFixture) block() *attnBlock {
	return &attnBlock{
		name:        "block",
		norm1Weight: fx.Norm1W, norm1Bias: fx.Norm1B, norm2Weight: fx.Norm2W, norm2Bias: fx.Norm2B,
		attention: tpaWeights{
			WAq: fx.WAq, WAk: fx.WAk, WAv: fx.WAv, WBq: fx.WBq, WBk: fx.WBk, WBv: fx.WBv,
			Woproj: fx.Woproj, Boproj: fx.Boproj, AlphaO: fx.AlphaO,
		},
		mlpProjection: fx.MLPGateW, mlpOutput: fx.MLPDownW,
		mlpProjected: 4 * fx.C, mlpHidden: 2 * fx.C,
		mlpAlpha: float32(fx.MLPAlpha), attentionScale: float32(fx.AttnScale), mlpScale: float32(fx.MLPScale),
	}
}

func (fx transformerBlockFixture) model() *Model {
	return &Model{Cfg: Config{
		Heads: fx.Heads, QRank: fx.QRank, KVRank: fx.KVRank,
		NormEps: fx.Eps, RopeTheta: vendorRopeTheta,
	}}
}

func TestTransformerBlockMatchesTorch(t *testing.T) {
	fx := loadGolden[transformerBlockFixture](t, "transformerblock")
	got := fx.block().forward(fx.model(), fx.X, fx.N, fx.C, fx.H, fx.W)
	requireWithin(t, "TransformerBlock", got, fx.Y, tolLoose)
}

type convProjectionFixture struct {
	B, CIn, COut, H, W, Kernel, Stride, OH, OW int
	Weight, Bias, X, Y                         []float32
}

type projectionOpsFixture struct {
	Patch, OneByOne, Transpose convProjectionFixture
}

func TestProjectionLeafOpsMatchTorch(t *testing.T) {
	fx := loadGolden[projectionOpsFixture](t, "projection_ops")
	for _, tc := range []struct {
		name string
		fx   convProjectionFixture
	}{{"patch Conv2d", fx.Patch}, {"1x1 Conv2d", fx.OneByOne}} {
		got, oh, ow := conv2dValidStride(tc.fx.X, tc.fx.Weight, tc.fx.Bias, tc.fx.B, tc.fx.CIn, tc.fx.COut, tc.fx.H, tc.fx.W, tc.fx.Kernel, tc.fx.Stride)
		if oh != tc.fx.OH || ow != tc.fx.OW {
			t.Fatalf("%s shape got %dx%d want %dx%d", tc.name, oh, ow, tc.fx.OH, tc.fx.OW)
		}
		requireWithin(t, tc.name, got, tc.fx.Y, tolTight)
	}
	got, oh, ow := convTranspose2dStride(fx.Transpose.X, fx.Transpose.Weight, fx.Transpose.Bias,
		fx.Transpose.B, fx.Transpose.CIn, fx.Transpose.COut, fx.Transpose.H, fx.Transpose.W, fx.Transpose.Kernel, fx.Transpose.Stride)
	if oh != fx.Transpose.OH || ow != fx.Transpose.OW {
		t.Fatalf("ConvTranspose shape got %dx%d want %dx%d", oh, ow, fx.Transpose.OH, fx.Transpose.OW)
	}
	requireWithin(t, "ConvTranspose", got, fx.Transpose.Y, tolTight)
}

func TestTinyModelForwardMatchesTorch(t *testing.T) {
	fx := loadGolden[uditTinyFixture](t, "udit_tiny")
	m := compileTiny(t, fx)
	got, err := m.Forward(fx.X, fx.B, fx.H, fx.W)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "tiny forward", got, fx.Y, tolLoose)
	// Trace twin must match the serving forward exactly.
	traced, _, err := m.forward(fx.X, fx.B, fx.H, fx.W, true)
	if err != nil {
		t.Fatal(err)
	}
	if d := testutil.MaxAbsDiff(got, traced); d != 0 {
		t.Fatalf("trace twin diverges from serving forward: max abs %g", d)
	}
}

func TestTokenReshapeRoundTrip(t *testing.T) {
	const b, c, h, w = 2, 3, 2, 2
	x := make([]float32, b*c*h*w)
	for i := range x {
		x[i] = float32(i)
	}
	back := tokensToImageNCHW(imageNCHWToTokens(x, b, c, h, w), b, c, h, w)
	if d := testutil.MaxAbsDiff(back, x); d != 0 {
		t.Fatalf("token reshape round trip diverges: %g", d)
	}
	if math.IsNaN(float64(back[0])) {
		t.Fatal("unreachable")
	}
}
