//go:build !amd64

package hostexpert

// avx2 is false off amd64: every format's rows take the scalar kernel.
const avx2 = false

var (
	iq4XSKernel = scalarKernel(iq4XSDot)
	iq4NLKernel = scalarKernel(iq4NLDot)
	iq3SKernel  = scalarKernel(iq3SDot)
	q80Kernel   = scalarKernel(q80Dot)
)
