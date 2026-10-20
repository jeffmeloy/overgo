//go:build !amd64

package hostexpert

var (
	iq4XSKernel = scalarKernel(iq4XSDot)
	iq4NLKernel = scalarKernel(iq4NLDot)
)
