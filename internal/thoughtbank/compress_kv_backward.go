package thoughtbank

import (
	"fmt"
	"math"
)

// CompressKVHeavyGradients mirrors CompressKVHeavy's inputs.
type CompressKVHeavyGradients struct {
	DHPad []float32 // [blocks*m, d_model]
	DWKV  []float32 // [d_head, d_model]
	DWZ   []float32 // [d_head, d_model]
	DPos  []float32 // [m, d_head]
}

// CompressKVHeavyBackward differentiates the HCA compression.
//
// Forward, per block b and feature e, over the m tokens in the block:
//
//	c[j,e] = W_kv[e] . tok_j
//	z[j,e] = W_z[e]  . tok_j + pos[j,e]
//	a[:,e] = softmax_j(z[:,e])          <- over the BLOCK axis, per feature
//	out[b,e] = sum_j a[j,e] * c[j,e]
//
// so with g = dL/dout[b,e]: dc[j,e]=g*a[j,e], da[j,e]=g*c[j,e], and dz couples
// through softmaxVJPInto. The max subtraction cancels between numerator and
// denominator here (unlike the sink softmax next door), so it contributes
// nothing to the gradient.
func CompressKVHeavyBackward(hPad, wKV, wZ, pos, dOut []float32,
	blocks, m, dModel, dHead int) (*CompressKVHeavyGradients, error) {
	if len(hPad) != blocks*m*dModel {
		return nil, fmt.Errorf("hca compress backward: input has %d values, want %d", len(hPad), blocks*m*dModel)
	}
	if len(pos) != m*dHead {
		return nil, fmt.Errorf("hca compress backward: pos has %d values, want %d", len(pos), m*dHead)
	}
	if len(dOut) != blocks*dHead {
		return nil, fmt.Errorf("hca compress backward: dOut has %d values, want %d", len(dOut), blocks*dHead)
	}

	g := &CompressKVHeavyGradients{
		DHPad: make([]float32, len(hPad)),
		DWKV:  make([]float32, len(wKV)),
		DWZ:   make([]float32, len(wZ)),
		DPos:  make([]float32, len(pos)),
	}
	c := make([]float64, m*dHead)
	a := make([]float64, m*dHead)
	pBuf := make([]float64, m)
	daBuf := make([]float64, m)
	dzBuf := make([]float64, m)

	for b := 0; b < blocks; b++ {
		// Recompute the forward's intermediates from the inputs; nothing carried
		// as stale forward state.
		z := make([]float64, m*dHead)
		for j := 0; j < m; j++ {
			tok := hPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				c[j*dHead+e] = dot(wKV[e*dModel:(e+1)*dModel], tok)
				z[j*dHead+e] = dot(wZ[e*dModel:(e+1)*dModel], tok) + float64(pos[j*dHead+e])
			}
		}
		for e := 0; e < dHead; e++ {
			maxZ := math.Inf(-1)
			for j := 0; j < m; j++ {
				if z[j*dHead+e] > maxZ {
					maxZ = z[j*dHead+e]
				}
			}
			sum := 0.0
			for j := 0; j < m; j++ {
				a[j*dHead+e] = math.Exp(z[j*dHead+e] - maxZ)
				sum += a[j*dHead+e]
			}
			for j := 0; j < m; j++ {
				a[j*dHead+e] /= sum
			}

			gv := float64(dOut[b*dHead+e])
			for j := 0; j < m; j++ {
				pBuf[j] = a[j*dHead+e]
				daBuf[j] = gv * c[j*dHead+e]
			}
			softmaxVJPInto(dzBuf, pBuf, daBuf)

			for j := 0; j < m; j++ {
				aj := a[j*dHead+e]
				dc := gv * aj
				dz := dzBuf[j]

				g.DPos[j*dHead+e] += float32(dz)

				tok := hPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
				dtok := g.DHPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
				wkvRow := wKV[e*dModel : (e+1)*dModel]
				wzRow := wZ[e*dModel : (e+1)*dModel]
				dkvRow := g.DWKV[e*dModel : (e+1)*dModel]
				dzRow := g.DWZ[e*dModel : (e+1)*dModel]
				for p := 0; p < dModel; p++ {
					dkvRow[p] += float32(dc * float64(tok[p]))
					dzRow[p] += float32(dz * float64(tok[p]))
					dtok[p] += float32(dc*float64(wkvRow[p]) + dz*float64(wzRow[p]))
				}
			}
		}
	}
	return g, nil
}

// CompressKVSparseGradients mirrors CompressKVSparse's inputs.
type CompressKVSparseGradients struct {
	DHPad []float32 // [blocks*m, d_model]
	DWKVa []float32 // [d_head, d_model]
	DWKVb []float32 // [d_head, d_model]
	DWZa  []float32 // [d_head, d_model]
	DWZb  []float32 // [d_head, d_model]
	DPosA []float32 // [m, d_head]
	DPosB []float32 // [m, d_head]
}

// CompressKVSparseBackward differentiates the CSA compression.
//
// It is the SAME operation the HCA pool differentiates -- a softmax-weighted sum
// of value projections over a candidate set, per feature -- sharing the coupling
// through softmaxVJPInto. The differences are structural: the candidate set is
// 2m wide (m from series a on this block, m from series b on the PREVIOUS
// block), so each candidate scatters into a different weight set and token span;
// block 0's phantom b-series has a -inf gate hence softmax weight 0 and carries
// no gradient (falls out of softmaxVJPInto for free), and is special-cased only
// to avoid indexing hPad at the negative span (b-1)*m. A token at (b*m+j) is
// read twice, so DHPad ACCUMULATES.
func CompressKVSparseBackward(hPad, wKVa, wKVb, wZa, wZb, posA, posB, dOut []float32,
	blocks, m, dModel, dHead int) (*CompressKVSparseGradients, error) {
	if len(hPad) != blocks*m*dModel {
		return nil, fmt.Errorf("csa compress backward: input has %d values, want %d", len(hPad), blocks*m*dModel)
	}
	if len(posA) != m*dHead || len(posB) != m*dHead {
		return nil, fmt.Errorf("csa compress backward: pos_a/pos_b must each hold %d values", m*dHead)
	}
	if len(dOut) != blocks*dHead {
		return nil, fmt.Errorf("csa compress backward: dOut has %d values, want %d", len(dOut), blocks*dHead)
	}

	g := &CompressKVSparseGradients{
		DHPad: make([]float32, len(hPad)),
		DWKVa: make([]float32, len(wKVa)),
		DWKVb: make([]float32, len(wKVb)),
		DWZa:  make([]float32, len(wZa)),
		DWZb:  make([]float32, len(wZb)),
		DPosA: make([]float32, len(posA)),
		DPosB: make([]float32, len(posB)),
	}
	cCat := make([]float64, 2*m*dHead)
	zCat := make([]float64, 2*m*dHead)
	a := make([]float64, 2*m*dHead)
	pBuf := make([]float64, 2*m)
	daBuf := make([]float64, 2*m)
	dzBuf := make([]float64, 2*m)

	for b := 0; b < blocks; b++ {
		for j := 0; j < m; j++ {
			tok := hPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				cCat[j*dHead+e] = dot(wKVa[e*dModel:(e+1)*dModel], tok)
				zCat[j*dHead+e] = dot(wZa[e*dModel:(e+1)*dModel], tok) + float64(posA[j*dHead+e])
			}
		}
		for j := 0; j < m; j++ {
			idx := (m + j) * dHead
			if b == 0 {
				for e := 0; e < dHead; e++ {
					cCat[idx+e] = 0
					zCat[idx+e] = math.Inf(-1)
				}
				continue
			}
			tok := hPad[((b-1)*m+j)*dModel : ((b-1)*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				cCat[idx+e] = dot(wKVb[e*dModel:(e+1)*dModel], tok)
				zCat[idx+e] = dot(wZb[e*dModel:(e+1)*dModel], tok) + float64(posB[j*dHead+e])
			}
		}

		for e := 0; e < dHead; e++ {
			maxZ := math.Inf(-1)
			for j := 0; j < 2*m; j++ {
				if v := zCat[j*dHead+e]; v > maxZ {
					maxZ = v
				}
			}
			sum := 0.0
			for j := 0; j < 2*m; j++ {
				a[j*dHead+e] = math.Exp(zCat[j*dHead+e] - maxZ)
				sum += a[j*dHead+e]
			}
			for j := 0; j < 2*m; j++ {
				a[j*dHead+e] /= sum
			}

			gv := float64(dOut[b*dHead+e])
			for j := 0; j < 2*m; j++ {
				pBuf[j] = a[j*dHead+e]
				daBuf[j] = gv * cCat[j*dHead+e]
			}
			softmaxVJPInto(dzBuf, pBuf, daBuf)

			for j := 0; j < 2*m; j++ {
				dc := gv * pBuf[j]
				dz := dzBuf[j]
				if j < m {
					// Series a on this block.
					g.DPosA[j*dHead+e] += float32(dz)
					off := (b*m + j) * dModel
					tok := hPad[off : off+dModel]
					dtok := g.DHPad[off : off+dModel]
					wkvRow := wKVa[e*dModel : (e+1)*dModel]
					wzRow := wZa[e*dModel : (e+1)*dModel]
					dkvRow := g.DWKVa[e*dModel : (e+1)*dModel]
					dzRow := g.DWZa[e*dModel : (e+1)*dModel]
					for p := 0; p < dModel; p++ {
						dkvRow[p] += float32(dc * float64(tok[p]))
						dzRow[p] += float32(dz * float64(tok[p]))
						dtok[p] += float32(dc*float64(wkvRow[p]) + dz*float64(wzRow[p]))
					}
					continue
				}
				// Series b on the PREVIOUS block; block 0's phantom carries none.
				if b == 0 {
					continue
				}
				jb := j - m
				g.DPosB[jb*dHead+e] += float32(dz)
				off := ((b-1)*m + jb) * dModel
				tok := hPad[off : off+dModel]
				dtok := g.DHPad[off : off+dModel]
				wkvRow := wKVb[e*dModel : (e+1)*dModel]
				wzRow := wZb[e*dModel : (e+1)*dModel]
				dkvRow := g.DWKVb[e*dModel : (e+1)*dModel]
				dzRow := g.DWZb[e*dModel : (e+1)*dModel]
				for p := 0; p < dModel; p++ {
					dkvRow[p] += float32(dc * float64(tok[p]))
					dzRow[p] += float32(dz * float64(tok[p]))
					dtok[p] += float32(dc*float64(wkvRow[p]) + dz*float64(wzRow[p]))
				}
			}
		}
	}
	return g, nil
}
