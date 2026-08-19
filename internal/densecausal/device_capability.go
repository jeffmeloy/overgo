package densecausal

// DeviceTrainingSupported decides, from model geometry alone, whether the
// resident device training backend can
// serve this model. The decision is host-side and is meant to run at SELECTION
// time -- before any device session is constructed -- so a model using a
// device-unsupported architecture trait is refused (or routed to host) up front
// instead of failing mid-session after the device is already open.
//
// It returns (true, "") when every trait the device backend needs is present.
func DeviceTrainingSupported(d Dims) (bool, string) {
	if d.MoE.TopK > 0 {
		return false, "routed mixture layers train on the host graph; the resident device lane has no mixture forward"
	}
	return true, ""
}
