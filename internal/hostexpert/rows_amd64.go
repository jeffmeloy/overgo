//go:build amd64

package hostexpert

// The AVX2 kernels (rows_amd64.s): a row's blocks against the activations of
// tokensPerKernel tokens or of one, Strata's iq_avx2 scheme. Each 32-value
// chunk of codes is looked up in the codebook once, its magnitudes multiplied
// by the activations carrying its signs, IQ4_XS's sub-block scale folded into
// the integer sums, and a block's float scales applied once a block.
//
//go:noescape
func iq4XSDots4(row *byte, blocks int, activations *[tokensPerKernel]*byte, results *[tokensPerKernel]float32)

//go:noescape
func iq4XSDots1(row *byte, blocks int, activation *byte) float32

//go:noescape
func iq4NLDots4(row *byte, blocks int, activations *[tokensPerKernel]*byte, results *[tokensPerKernel]float32)

//go:noescape
func iq4NLDots1(row *byte, blocks int, activation *byte) float32

// cpuid and xgetbv read the processor's features and the state the system
// saves for it.
func cpuid(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)

func xgetbv() (eax, edx uint32)

// The feature bits the kernels need: CPUID leaf 1's FMA, OSXSAVE, AVX and
// F16C in ECX; leaf 7's AVX2 in EBX; and XCR0's SSE and AVX state.
const (
	leafFeatures         = 1
	leafExtendedFeatures = 7
	ecxFMA               = 1 << 12
	ecxOSXSAVE           = 1 << 27
	ecxAVX               = 1 << 28
	ecxF16C              = 1 << 29
	ebxAVX2              = 1 << 5
	xcr0SSEAVX           = 0b110
)

// avx2 reports whether the processor and the system run the kernels.
var avx2 = func() bool {
	if maximum, _, _, _ := cpuid(0, 0); maximum < leafExtendedFeatures {
		return false
	}
	_, _, features, _ := cpuid(leafFeatures, 0)
	needed := uint32(ecxFMA | ecxOSXSAVE | ecxAVX | ecxF16C)
	if features&needed != needed {
		return false
	}
	if state, _ := xgetbv(); state&xcr0SSEAVX != xcr0SSEAVX {
		return false
	}
	_, extended, _, _ := cpuid(leafExtendedFeatures, 0)
	return extended&ebxAVX2 != 0
}()

var (
	iq4XSKernel = vectorKernel(iq4XSDots4, iq4XSDots1, iq4XSDot)
	iq4NLKernel = vectorKernel(iq4NLDots4, iq4NLDots1, iq4NLDot)
)

// vectorKernel is the AVX2 kernel where the processor runs it and the scalar
// formula elsewhere. A group of fewer than tokensPerKernel tokens of more
// than one repeats its first token in the empty places.
func vectorKernel(four func(*byte, int, *[tokensPerKernel]*byte, *[tokensPerKernel]float32), one func(*byte, int, *byte) float32,
	dot func(row, activations []byte, blocks int) float32) rowKernel {
	if !avx2 {
		return scalarKernel(dot)
	}
	return func(row []byte, blocks int, activations [][]byte, results []float32) {
		if len(activations) == 1 {
			results[0] = one(&row[0], blocks, &activations[0][0])
			return
		}
		var pointers [tokensPerKernel]*byte
		for index := range pointers {
			pointers[index] = &activations[min(index, len(activations)-1)][0]
		}
		var products [tokensPerKernel]float32
		four(&row[0], blocks, &pointers, &products)
		copy(results, products[:len(activations)])
	}
}
