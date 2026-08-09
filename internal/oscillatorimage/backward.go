package oscillatorimage

import "math"

// Host VJPs for the training leg. Gate: artifact-shipped torch autograd
// goldens for conv2dSame3x3 and resizeConvBlock; central differences for the
// dynamics and readout operators; a descending drift loss end to end.

// conv2dSame3x3Backward: (dx, dWeight, dBias); f64 accumulation.
func conv2dSame3x3Backward(x, weight, dOut []float32, b, cin, cout, h, w int) (dx, dW, dB []float32) {
	plane := h * w
	dxf := make([]float64, b*cin*plane)
	dWf := make([]float64, cout*cin*convTaps)
	dBf := make([]float64, cout)
	for bi := 0; bi < b; bi++ {
		for co := 0; co < cout; co++ {
			gb := (bi*cout + co) * plane
			for _, g := range dOut[gb : gb+plane] {
				dBf[co] += float64(g)
			}
			for ci := 0; ci < cin; ci++ {
				xb := (bi*cin + ci) * plane
				wc := (co*cin + ci) * convTaps
				for i := 0; i < h; i++ {
					for j := 0; j < w; j++ {
						g := float64(dOut[gb+i*w+j])
						if g == 0 {
							continue
						}
						for di := -1; di <= 1; di++ {
							si := i + di
							if si < 0 || si >= h {
								continue
							}
							for dj := -1; dj <= 1; dj++ {
								sj := j + dj
								if sj < 0 || sj >= w {
									continue
								}
								wi := wc + (di+1)*convKernel + (dj + 1)
								dWf[wi] += g * float64(x[xb+si*w+sj])
								dxf[xb+si*w+sj] += g * float64(weight[wi])
							}
						}
					}
				}
			}
		}
	}
	return f64ToF32(dxf), f64ToF32(dWf), f64ToF32(dBf)
}

// upsampleNearest2xBackward: 2x2 sum-pool VJP.
func upsampleNearest2xBackward(dOut []float32, b, c, h, w int) []float32 {
	oh, ow := upsample*h, upsample*w
	dx := make([]float32, b*c*h*w)
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			ob := (bi*c + ci) * oh * ow
			ib := (bi*c + ci) * h * w
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					dx[ib+(i/upsample)*w+(j/upsample)] += dOut[ob+i*ow+j]
				}
			}
		}
	}
	return dx
}

// leakyReLUBackwardInto: VJP against the pre-activation; in-place dOut valid.
func leakyReLUBackwardInto(d, preact, dOut []float32, slope float64) {
	s := float32(slope)
	for i := range dOut {
		if preact[i] < 0 {
			d[i] = s * dOut[i]
		} else {
			d[i] = dOut[i]
		}
	}
}

// resizeConvBlockBackward: recompute forward intermediates, chain the
// leaky/conv/leaky/conv/upsample VJPs in reverse.
func resizeConvBlockBackward(x, w1, b1, w2, b2, dOut []float32, b, cin, cout, h, w int, slope float64) (dx, dW1, dB1, dW2, dB2 []float32) {
	oh, ow := upsample*h, upsample*w
	up := upsampleNearest2x(x, b, cin, h, w)
	c1pre := conv2dSame3x3(up, w1, b1, b, cin, cout, oh, ow)
	c1 := make([]float32, len(c1pre))
	copy(c1, c1pre)
	leakyReLU(c1, slope)
	c2pre := conv2dSame3x3(c1, w2, b2, b, cout, cout, oh, ow)
	dC2pre := make([]float32, len(dOut))
	leakyReLUBackwardInto(dC2pre, c2pre, dOut, slope)
	dC1, dW2, dB2 := conv2dSame3x3Backward(c1, w2, dC2pre, b, cout, cout, oh, ow)
	leakyReLUBackwardInto(dC1, c1pre, dC1, slope)
	dUp, dW1, dB1 := conv2dSame3x3Backward(up, w1, dC1, b, cin, cout, oh, ow)
	dx = upsampleNearest2xBackward(dUp, b, cin, h, w)
	return dx, dW1, dB1, dW2, dB2
}

// kuramotoVelocityBackwardAccumulate: one-group velocity VJP. dTheta rows are
// REPLACED; dOmega/dK accumulate (time steps sum into the same parameters).
func kuramotoVelocityBackwardAccumulate(dTheta []float32, dThetaStride, dThetaOffset int, dOmega, dK []float32, theta []float32, thetaStride, thetaOffset int, coupling, dVel []float32, dVelStride, dVelOffset, b, n int, scale float64, zeroDiagonal bool) {
	dOmegaF := make([]float64, n)
	dKf := make([]float64, n*n)
	sinT, cosT := make([]float64, n), make([]float64, n)
	ws, wc := make([]float64, n), make([]float64, n)
	for bi := 0; bi < b; bi++ {
		th := theta[bi*thetaStride+thetaOffset:][:n]
		dv := dVel[bi*dVelStride+dVelOffset:][:n]
		for j := 0; j < n; j++ {
			sinT[j], cosT[j] = math.Sincos(float64(th[j]))
		}
		clear(ws)
		clear(wc)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if zeroDiagonal && i == j {
					continue
				}
				k := float64(float32(float64(coupling[i*n+j]) * scale))
				ws[i] += k * sinT[j]
				wc[i] += k * cosT[j]
			}
		}
		for i := 0; i < n; i++ {
			dOmegaF[i] += float64(dv[i])
			for j := 0; j < n; j++ {
				if !zeroDiagonal || i != j {
					dKf[i*n+j] += float64(dv[i]) * (sinT[j]*cosT[i] - cosT[j]*sinT[i])
				}
			}
		}
		for k := 0; k < n; k++ {
			d := float64(dv[k]) * (-sinT[k]*ws[k] - cosT[k]*wc[k])
			for i := 0; i < n; i++ {
				if zeroDiagonal && i == k {
					continue
				}
				couplingValue := float64(float32(float64(coupling[i*n+k]) * scale))
				d += float64(dv[i]) * couplingValue * (cosT[i]*cosT[k] + sinT[i]*sinT[k])
			}
			dTheta[bi*dThetaStride+dThetaOffset+k] = float32(d)
		}
	}
	for i := 0; i < n; i++ {
		dOmega[i] += float32(dOmegaF[i])
	}
	for i := 0; i < n*n; i++ {
		dK[i] += float32(float64(float32(dKf[i])) * scale)
	}
}

// conditionalKuramotoBackwardAccumulate: coupled phase VJP. dState is
// REPLACED; parameter grads accumulate.
func conditionalKuramotoBackwardAccumulate(dState, dOmega, dK, dOmegaCond, dKCond, dDrive, state, kMat, kCondMat, drive, dOut []float32, b, n, nCond int, kScale, kCondScale, kDriveScale float64) {
	tot := n + nCond
	clear(dState)
	kuramotoVelocityBackwardAccumulate(dState, tot, 0, dOmega, dK, state, tot, 0, kMat, dOut, tot, 0, b, n, kScale, true)
	kuramotoVelocityBackwardAccumulate(dState, tot, n, dOmegaCond, dKCond, state, tot, n, kCondMat, dOut, tot, n, b, nCond, kCondScale, true)
	sinC, cosC := make([]float64, nCond), make([]float64, nCond)
	for bi := 0; bi < b; bi++ {
		for m := 0; m < nCond; m++ {
			sinC[m], cosC[m] = math.Sincos(float64(state[bi*tot+n+m]))
		}
		for i := 0; i < n; i++ {
			sm, cm := math.Sincos(float64(state[bi*tot+i]))
			var ds, dc float64
			base := bi*n*nCond + i*nCond
			for m := 0; m < nCond; m++ {
				de := float64(drive[base+m]) * kDriveScale
				ds += de * sinC[m]
				dc += de * cosC[m]
			}
			dmv := float64(dOut[bi*tot+i])
			dState[bi*tot+i] = float32(float64(dState[bi*tot+i]) + dmv*(-sm*ds-cm*dc))
			for m := 0; m < nCond; m++ {
				de := float64(drive[base+m]) * kDriveScale
				dState[bi*tot+n+m] = float32(float64(dState[bi*tot+n+m]) + dmv*de*(cm*cosC[m]+sm*sinC[m]))
				dDrive[base+m] += float32(dmv * (cm*sinC[m] - sm*cosC[m]) * kDriveScale)
			}
		}
	}
}

// readoutTransformBackwardInto: phase readout VJP.
func readoutTransformBackwardInto(dPhases []float32, dStride, dOffset int, dFeat, phases []float32, b, n, phaseStride, phaseOffset int, relativization, encoding string) {
	dp := make([]float64, n)
	for bi := 0; bi < b; bi++ {
		row := phases[bi*phaseStride+phaseOffset:][:n]
		var mean float64
		if relativization == "mean_relative" {
			for _, value := range row {
				mean += float64(value)
			}
			mean /= float64(n)
		}
		ref := float32(0)
		if relativization == "ref_oscillator" {
			ref = row[0]
		}
		for j, value := range row {
			if relativization == "mean_relative" {
				value = float32(float64(value) - mean)
			} else {
				value -= ref
			}
			switch encoding {
			case "sin":
				dp[j] = float64(dFeat[bi*n+j]) * math.Cos(float64(value))
			case "sin_cos":
				s, c := math.Sincos(float64(value))
				dp[j] = float64(dFeat[bi*2*n+j])*c - float64(dFeat[bi*2*n+n+j])*s
			default:
				dp[j] = float64(dFeat[bi*n+j])
			}
		}
		out := dPhases[bi*dStride+dOffset:][:n]
		switch relativization {
		case "ref_oscillator":
			var sum float64
			for _, value := range dp {
				sum += value
			}
			for j := 0; j < n; j++ {
				v := dp[j]
				if j == 0 {
					v -= sum
				}
				out[j] = float32(v)
			}
		case "mean_relative":
			var sum float64
			for _, value := range dp {
				sum += value
			}
			for j := 0; j < n; j++ {
				out[j] = float32(dp[j] - sum/float64(n))
			}
		default:
			for j := 0; j < n; j++ {
				out[j] = float32(dp[j])
			}
		}
	}
}

func f64ToF32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = float32(v)
	}
	return out
}
