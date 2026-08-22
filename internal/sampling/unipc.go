// Flow-matching UniPC sampling boundary: shifted flow schedule, the
// order-2 predictor/corrector update, classifier-free guidance, and the
// Philox counter-based normal noise source. Pure host math — each element is
// one double-precision expression rounded once to float32, replicating the
// reference engine's device kernels exactly.
package sampling

import (
	"fmt"
	"math"

	"overgo/internal/checked"
)

// UniPCSchedule: shifted flow schedule. Returns inferenceSteps timesteps and
// inferenceSteps+1 sigmas; the trailing sigma is the terminal zero.
func UniPCSchedule(numTrainTimesteps, inferenceSteps int, shift float64) ([]int64, []float32, error) {
	if numTrainTimesteps <= 0 || inferenceSteps <= 0 {
		return nil, nil, fmt.Errorf("unipc schedule: timesteps/steps must be positive, got %d/%d", numTrainTimesteps, inferenceSteps)
	}
	if shift <= 0 || math.IsNaN(shift) || math.IsInf(shift, 0) {
		return nil, nil, fmt.Errorf("unipc schedule: shift must be finite and positive, got %g", shift)
	}
	sigmaMax := 1 - 1/float64(numTrainTimesteps)
	timesteps := make([]int64, inferenceSteps)
	sigmas := make([]float32, inferenceSteps+1)
	for i := 0; i < inferenceSteps; i++ {
		base := sigmaMax * (1 - float64(i)/float64(inferenceSteps))
		sigma := shift * base / (1 + (shift-1)*base)
		timesteps[i] = int64(sigma * float64(numTrainTimesteps))
		sigmas[i] = float32(sigma)
	}
	return timesteps, sigmas, nil
}

// UniPCSampler: order-2 flow UniPC predictor/corrector over host buffers.
// State layout and buffer rotation mirror the reference device scheduler.
type UniPCSampler struct {
	sigmas         []float32
	stepIndex      int
	lowerOrderNums int
	thisOrder      int
	modelPrevious  []float32
	modelCurrent   []float32
	modelCandidate []float32
	lastSample     []float32
	previousValid  bool
	currentValid   bool
}

func NewUniPCSampler(sigmas []float32, elements int) (*UniPCSampler, error) {
	if elements <= 0 {
		return nil, fmt.Errorf("unipc sampler: invalid elements=%d", elements)
	}
	if err := ValidateSigmaSchedule(sigmas); err != nil {
		return nil, fmt.Errorf("unipc sampler: %w", err)
	}
	return &UniPCSampler{
		sigmas:         append([]float32(nil), sigmas...),
		modelPrevious:  make([]float32, elements),
		modelCurrent:   make([]float32, elements),
		modelCandidate: make([]float32, elements),
		lastSample:     make([]float32, elements),
	}, nil
}

// Step: consumes the model output at the current sample and writes the next
// sample into output. Buffers may not alias.
func (p *UniPCSampler) Step(output, sample, modelOutput []float32) error {
	if p == nil || len(p.sigmas) < 2 {
		return fmt.Errorf("unipc sampler is nil")
	}
	elements := len(p.lastSample)
	if len(output) != elements || len(sample) != elements || len(modelOutput) != elements {
		return fmt.Errorf("unipc sampler: buffer lengths %d/%d/%d differ from %d", len(output), len(sample), len(modelOutput), elements)
	}
	if p.stepIndex >= len(p.sigmas)-1 {
		return fmt.Errorf("unipc sampler: step index %d past schedule", p.stepIndex)
	}
	// Convert to the data prediction: candidate = sample - sigma*model.
	sigma := float64(p.sigmas[p.stepIndex])
	for i := range p.modelCandidate {
		p.modelCandidate[i] = float32(float64(sample[i]) - sigma*float64(modelOutput[i]))
	}
	workingSample := sample
	if p.stepIndex > 0 && p.currentValid {
		if p.thisOrder <= 1 || !p.previousValid {
			c0, c1, c2 := p.uniCOrder1Coeffs(p.stepIndex, p.stepIndex-1)
			for i := range p.lastSample {
				av, bv, cv := float64(p.lastSample[i]), float64(p.modelCurrent[i]), float64(p.modelCandidate[i])
				p.lastSample[i] = float32(c0*av - c1*bv - c2*(cv-bv))
			}
		} else {
			c0, c1, c2, c3 := p.uniCOrder2Coeffs(p.stepIndex, p.stepIndex-1, p.stepIndex-2)
			for i := range p.lastSample {
				av, bv := float64(p.lastSample[i]), float64(p.modelCurrent[i])
				cv, dv := float64(p.modelCandidate[i]), float64(p.modelPrevious[i])
				p.lastSample[i] = float32(c0*av - c1*bv - c2*(cv-bv) - c3*(dv-bv))
			}
		}
		workingSample = p.lastSample
	}
	if p.currentValid {
		p.modelPrevious, p.modelCurrent, p.modelCandidate = p.modelCurrent, p.modelCandidate, p.modelPrevious
		p.previousValid = true
	} else {
		p.modelCurrent, p.modelCandidate = p.modelCandidate, p.modelCurrent
	}
	p.currentValid = true
	thisOrder := min(2, len(p.sigmas)-1-p.stepIndex)
	p.thisOrder = min(thisOrder, p.lowerOrderNums+1)
	if p.thisOrder <= 0 {
		return fmt.Errorf("unipc sampler: invalid order %d", p.thisOrder)
	}
	if &workingSample[0] != &p.lastSample[0] {
		copy(p.lastSample, workingSample)
	}
	if p.thisOrder == 1 || !p.previousValid {
		c0, c1 := p.uniPOrder1Coeffs(p.stepIndex, p.stepIndex+1)
		for i := range output {
			output[i] = float32(c0*float64(p.lastSample[i]) - c1*float64(p.modelCurrent[i]))
		}
	} else {
		c0, c1, c2 := p.uniPOrder2Coeffs(p.stepIndex, p.stepIndex+1, p.stepIndex-1)
		for i := range output {
			av, bv, cv := float64(p.lastSample[i]), float64(p.modelCurrent[i]), float64(p.modelPrevious[i])
			output[i] = float32(c0*av - c1*bv - c2*(cv-bv))
		}
	}
	if p.lowerOrderNums < 2 {
		p.lowerOrderNums++
	}
	p.stepIndex++
	return nil
}

func (p *UniPCSampler) uniPOrder1Coeffs(s0, t int) (sampleScale, modelScale float64) {
	sigmaCur, sigmaNext := float64(p.sigmas[s0]), float64(p.sigmas[t])
	if checked.Equal(sigmaNext, float64(0)) {
		return 0, -1
	}
	alphaCur, alphaNext := 1-sigmaCur, 1-sigmaNext
	h := math.Log(alphaNext) - math.Log(sigmaNext) - math.Log(alphaCur) + math.Log(sigmaCur)
	return sigmaNext / sigmaCur, alphaNext * math.Expm1(-h)
}

func (p *UniPCSampler) uniPOrder2Coeffs(s0, t, previous int) (sampleScale, modelScale, deltaScale float64) {
	sigmaT, sigmaS0 := float64(p.sigmas[t]), float64(p.sigmas[s0])
	if checked.Equal(sigmaT, float64(0)) {
		return 0, -1, 0
	}
	alphaT := 1 - sigmaT
	lambdaT := math.Log(alphaT) - math.Log(sigmaT)
	lambdaS0 := math.Log(1-sigmaS0) - math.Log(sigmaS0)
	h := lambdaT - lambdaS0
	sigmaPrevious := float64(p.sigmas[previous])
	lambdaPrevious := math.Log(1-sigmaPrevious) - math.Log(sigmaPrevious)
	rk := (lambdaPrevious - lambdaS0) / h
	hPhi1 := math.Expm1(-h)
	return sigmaT / sigmaS0, alphaT * hPhi1, alphaT * hPhi1 * 0.5 / rk
}

func (p *UniPCSampler) uniCOrder1Coeffs(t, s0 int) (lastScale, modelScale, deltaScale float64) {
	sigmaT, sigmaS0 := float64(p.sigmas[t]), float64(p.sigmas[s0])
	alphaT := 1 - sigmaT
	h := math.Log(alphaT) - math.Log(sigmaT) - math.Log(1-sigmaS0) + math.Log(sigmaS0)
	hPhi1 := math.Expm1(-h)
	return sigmaT / sigmaS0, alphaT * hPhi1, alphaT * hPhi1 * 0.5
}

func (p *UniPCSampler) uniCOrder2Coeffs(t, s0, previous int) (lastScale, modelScale, candidateScale, previousScale float64) {
	sigmaT, sigmaS0 := float64(p.sigmas[t]), float64(p.sigmas[s0])
	alphaT := 1 - sigmaT
	lambdaT := math.Log(alphaT) - math.Log(sigmaT)
	lambdaS0 := math.Log(1-sigmaS0) - math.Log(sigmaS0)
	h := lambdaT - lambdaS0
	sigmaPrevious := float64(p.sigmas[previous])
	lambdaPrevious := math.Log(1-sigmaPrevious) - math.Log(sigmaPrevious)
	rk := (lambdaPrevious - lambdaS0) / h
	hh := -h
	hPhi1 := math.Expm1(hh)
	hPhiK := hPhi1/hh - 1
	b0 := hPhiK / hPhi1
	hPhiK = hPhiK/hh - 0.5
	b1 := 2 * hPhiK / hPhi1
	rho0 := (b1 - b0) / (rk - 1)
	rho1 := b0 - rho0
	return sigmaT / sigmaS0, alphaT * hPhi1, alphaT * hPhi1 * rho1, alphaT * hPhi1 * rho0 / rk
}

// GuideInto: classifier-free guidance out = uncond + guide*(cond - uncond).
func GuideInto(out, conditional, unconditional []float32, guide float64) error {
	if len(out) != len(conditional) || len(out) != len(unconditional) {
		return fmt.Errorf("guidance: buffer lengths %d/%d/%d differ", len(out), len(conditional), len(unconditional))
	}
	for i := range out {
		c, u := float64(conditional[i]), float64(unconditional[i])
		out[i] = float32(u + guide*(c-u))
	}
	return nil
}
