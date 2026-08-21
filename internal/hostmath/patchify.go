// Patchify layout kernels: channel-major latent grids to token-major patch
// rows and back, plus the exact transposes (the VJP scatters — both maps are
// permutations, so each transpose is the inverse index map). Single owner of
// the diffusion patch layout; capability packages supply every dimension.
package hostmath

// PatchifyChannelMajor: channel-major latent [channels, frames, height,
// width] -> token-major patch rows [seq, channels*p0*p1*p2] with
// seq = (frames/p0)*(height/p1)*(width/p2) and row layout
// ((channel*p0+kt)*p1+kh)*p2+kw. dst and src must not alias.
func PatchifyChannelMajor(dst, src []float32, channels, frames, height, width, p0, p1, p2 int) {
	patchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2, func(rowIndex, latentIndex int) {
		dst[rowIndex] = src[latentIndex]
	})
}

// The patchify transpose is deliberately absent: the latent input is data
// (x_t), so no trainer backpropagates through PatchifyChannelMajor; the
// bijection's adjoint is pinned in the tests via the shared index map.

func patchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2 int, visit func(rowIndex, latentIndex int)) {
	grid1, grid2 := height/p1, width/p2
	seq := frames / p0 * grid1 * grid2
	patchIn := channels * p0 * p1 * p2
	for token := 0; token < seq; token++ {
		ft := token / (grid1 * grid2)
		remainder := token % (grid1 * grid2)
		hy, wx := remainder/grid2, remainder%grid2
		base := token * patchIn
		for channel := 0; channel < channels; channel++ {
			for kt := 0; kt < p0; kt++ {
				for kh := 0; kh < p1; kh++ {
					for kw := 0; kw < p2; kw++ {
						latentIndex := (((channel*frames+ft*p0+kt)*height + hy*p1 + kh) * width) + wx*p2 + kw
						visit(base+((channel*p0+kt)*p1+kh)*p2+kw, latentIndex)
					}
				}
			}
		}
	}
}

// UnpatchifyChannelMajor: token-major head rows [seq, p0*p1*p2*channels]
// (vector layout ((kt*p1+kh)*p2+kw)*channels + channel) -> channel-major
// latent [channels, frames, height, width] with frames = (seq/(grid1*grid2))*p0.
func UnpatchifyChannelMajor(dst, src []float32, channels, frames, height, width, p0, p1, p2 int) {
	unpatchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2, func(rowIndex, latentIndex int) {
		dst[latentIndex] = src[rowIndex]
	})
}

// UnpatchifyChannelMajorTranspose: exact transpose of UnpatchifyChannelMajor
// — the latent gradient gathers back into the token-row gradient.
func UnpatchifyChannelMajorTranspose(dRows, dLatent []float32, channels, frames, height, width, p0, p1, p2 int) {
	unpatchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2, func(rowIndex, latentIndex int) {
		dRows[rowIndex] = dLatent[latentIndex]
	})
}

func unpatchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2 int, visit func(rowIndex, latentIndex int)) {
	grid1, grid2 := height/p1, width/p2
	seq := frames / p0 * grid1 * grid2
	patchOut := channels * p0 * p1 * p2
	for token := 0; token < seq; token++ {
		frame := token / (grid1 * grid2)
		remainder := token % (grid1 * grid2)
		row, column := remainder/grid2, remainder%grid2
		base := token * patchOut
		for kt := 0; kt < p0; kt++ {
			for kh := 0; kh < p1; kh++ {
				for kw := 0; kw < p2; kw++ {
					for channel := 0; channel < channels; channel++ {
						latentIndex := (((channel*frames+frame*p0+kt)*height + row*p1 + kh) * width) + column*p2 + kw
						visit(base+((kt*p1+kh)*p2+kw)*channels+channel, latentIndex)
					}
				}
			}
		}
	}
}
