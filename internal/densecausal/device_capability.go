package densecausal

// DeviceTrainingSupported decides, from model geometry alone, whether the
// resident device training backend (TrainDeviceFull / TrainDeviceResident) can
// serve this model. The decision is host-side and is meant to run at SELECTION
// time -- before any device session is constructed -- so a model using a
// device-unsupported architecture trait is refused (or routed to host) up front
// instead of failing mid-session after the device is already open.
//
// It returns (true, "") when every trait the device backend needs is present,
// or (false, reason) naming the first unsupported trait. Attention bias is the
// current gap: the resident layer forward/backward and deviceLossAndGrads carry
// no bias term. New device-unsupported traits belong in deviceTrainingLimits,
// not as ad-hoc checks scattered across call sites.
func DeviceTrainingSupported(d Dims) (bool, string) {
	for _, limit := range deviceTrainingLimits {
		if limit.unsupported(d) {
			return false, limit.reason
		}
	}
	return true, ""
}

// deviceTrainingLimit is one architecture trait the device training backend
// cannot serve yet. The extension point for future gaps: append one entry and
// every selection site refuses (or reroutes) the trait uniformly.
type deviceTrainingLimit struct {
	reason      string
	unsupported func(Dims) bool
}

// deviceTrainingLimits is the single owner of "what the device training backend
// cannot do". The device forward/backward guards consult it too, so the reason
// text lives in exactly one place.
var deviceTrainingLimits = []deviceTrainingLimit{
	{
		reason:      "attention bias (q/k/v projection bias) is not supported by the device training backend",
		unsupported: func(d Dims) bool { return d.AttnBias },
	},
}
