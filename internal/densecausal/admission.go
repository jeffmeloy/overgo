package densecausal

// DeviceTrainingAdmitted reports whether the device training path
// (TrainDeviceFull) supports this model, with a human-readable reason when it
// does not. Backend selection must consult this rather than assume the device
// path handles every loadable model: the device backward (deviceLayerBackward)
// does not yet implement attention-bias gradients, so bias models — e.g. Qwen2,
// which requires q/k/v biases — are not admitted and must train on the host
// path. Keep this in lockstep with deviceLayerBackward's constraints.
func (m *Model) DeviceTrainingAdmitted() (bool, string) {
	if m.Dims.AttnBias {
		return false, "device backward does not support attention bias yet"
	}
	return true, ""
}
