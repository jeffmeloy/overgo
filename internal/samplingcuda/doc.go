// Package samplingcuda takes samplers' steps on the card, each as its host step in internal/sampling takes it, bit
// for bit, so a sampling keeps its latent there (moss-step-copies).
package samplingcuda
