//go:build windows

package cublas

import (
	"errors"
	"syscall"
	"unsafe"

	"overgo/internal/cuda/driver"
)

// cuBLASLt enumerants the FP8 product sets (cublasLt.h, CUDA 12.9).
const (
	// dataF8E4M3 is CUDA_R_8F_E4M3: OCP E4M3FN bytes.
	dataF8E4M3 = DataType(28)
	// ltDescTransA and ltDescTransB: CUBLASLT_MATMUL_DESC_TRANSA/B, int32.
	ltDescTransA = uintptr(3)
	ltDescTransB = uintptr(4)
	// ltPreferenceMaxWorkspace: CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, uint64.
	ltPreferenceMaxWorkspace = uintptr(1)
	// ltHeuristicTailBytes: the fields of cublasLtMatmulHeuristicResult_t
	// after the algorithm (size_t workspaceSize, int state, float
	// wavesCount, int reserved[4]), which the plan does not read.
	ltHeuristicTailBytes = 32
	// statusNotSupported: CUBLAS_STATUS_NOT_SUPPORTED.
	statusNotSupported = int32(15)
)

// ltHeuristicResult mirrors cublasLtMatmulHeuristicResult_t.
type ltHeuristicResult struct {
	algo [ltAlgoWords]uint64
	_    [ltHeuristicTailBytes]byte
}

// ltProcs: the cuBLASLt entry points the FP8 product uses.
type ltProcs struct {
	create, destroy                                             *syscall.Proc
	descCreate, descDestroy, descSetAttribute                   *syscall.Proc
	layoutCreate, layoutDestroy                                 *syscall.Proc
	preferenceCreate, preferenceDestroy, preferenceSetAttribute *syscall.Proc
	heuristic, matmul                                           *syscall.Proc
}

func (p *ltProcs) bindings() []procBinding {
	return []procBinding{
		{"cublasLtCreate", &p.create},
		{"cublasLtDestroy", &p.destroy},
		{"cublasLtMatmulDescCreate", &p.descCreate},
		{"cublasLtMatmulDescDestroy", &p.descDestroy},
		{"cublasLtMatmulDescSetAttribute", &p.descSetAttribute},
		{"cublasLtMatrixLayoutCreate", &p.layoutCreate},
		{"cublasLtMatrixLayoutDestroy", &p.layoutDestroy},
		{"cublasLtMatmulPreferenceCreate", &p.preferenceCreate},
		{"cublasLtMatmulPreferenceDestroy", &p.preferenceDestroy},
		{"cublasLtMatmulPreferenceSetAttribute", &p.preferenceSetAttribute},
		{"cublasLtMatmulAlgoGetHeuristic", &p.heuristic},
		{"cublasLtMatmul", &p.matmul},
	}
}

// CreateLt creates a cuBLASLt context.
func (l *Library) CreateLt() (LtHandle, error) {
	var handle LtHandle
	status, _, _ := l.lt.create.Call(uintptr(unsafe.Pointer(&handle)))
	if err := result("cublasLtCreate", status); err != nil {
		return 0, err
	}
	return handle, nil
}

// DestroyLt releases a cuBLASLt context.
func (l *Library) DestroyLt(handle LtHandle) error {
	if handle == 0 {
		return nil
	}
	status, _, _ := l.lt.destroy.Call(uintptr(handle))
	return result("cublasLtDestroy", status)
}

// PlanFP8 builds the descriptors of an m x n x k FP8 product (see
// FP8Plan) and takes cuBLASLt's first heuristic algorithm needing at
// most workspaceLimit bytes of workspace. plan owns every descriptor
// created until the product is built: an error return overwrites the
// result, never plan, so a failed construction destroys each one.
func (l *Library) PlanFP8(handle LtHandle, m, n, k, workspaceLimit uint64) (built *FP8Plan, err error) {
	plan := &FP8Plan{}
	defer func() {
		if err != nil {
			err = errors.Join(err, l.DestroyFP8Plan(plan))
			built = nil
		}
	}()
	status, _, _ := l.lt.descCreate.Call(uintptr(unsafe.Pointer(&plan.desc)), uintptr(ComputeF32), uintptr(DataF32))
	if err := result("cublasLtMatmulDescCreate", status); err != nil {
		return nil, err
	}
	for _, attribute := range []struct {
		id    uintptr
		value int32
	}{
		{ltDescTransA, int32(OperationTranspose)},
		{ltDescTransB, int32(OperationNone)},
	} {
		if err := setAttribute(l.lt.descSetAttribute, "cublasLtMatmulDescSetAttribute", plan.desc, attribute.id, attribute.value); err != nil {
			return nil, err
		}
	}
	// Every operand is column-major with its leading dimension equal to
	// its row count: A is k x m (op T), B is k x n, D is m x n.
	for _, layout := range []struct {
		dst        *uintptr
		dataType   DataType
		rows, cols uint64
	}{
		{&plan.a, dataF8E4M3, k, m},
		{&plan.b, dataF8E4M3, k, n},
		{&plan.d, DataF32, m, n},
	} {
		status, _, _ = l.lt.layoutCreate.Call(
			uintptr(unsafe.Pointer(layout.dst)), uintptr(layout.dataType),
			uintptr(layout.rows), uintptr(layout.cols), uintptr(layout.rows),
		)
		if err := result("cublasLtMatrixLayoutCreate", status); err != nil {
			return nil, err
		}
	}
	var preference uintptr
	status, _, _ = l.lt.preferenceCreate.Call(uintptr(unsafe.Pointer(&preference)))
	if err := result("cublasLtMatmulPreferenceCreate", status); err != nil {
		return nil, err
	}
	defer func() {
		status, _, _ := l.lt.preferenceDestroy.Call(preference)
		err = errors.Join(err, result("cublasLtMatmulPreferenceDestroy", status))
	}()
	if err := setAttribute(l.lt.preferenceSetAttribute, "cublasLtMatmulPreferenceSetAttribute", preference, ltPreferenceMaxWorkspace, workspaceLimit); err != nil {
		return nil, err
	}
	var chosen ltHeuristicResult
	var count int32
	status, _, _ = l.lt.heuristic.Call(
		uintptr(handle), plan.desc, plan.a, plan.b, plan.d, plan.d, preference,
		1, uintptr(unsafe.Pointer(&chosen)), uintptr(unsafe.Pointer(&count)),
	)
	if err := result("cublasLtMatmulAlgoGetHeuristic", status); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, &StatusError{Operation: "cublasLtMatmulAlgoGetHeuristic", Status: statusNotSupported}
	}
	plan.algo = chosen.algo
	return plan, nil
}

// MatmulFP8 enqueues plan's product d = a^T*b on stream, in a
// workspace of at least the limit the plan was chosen under.
func (l *Library) MatmulFP8(
	handle LtHandle,
	plan *FP8Plan,
	a, b, d, workspace driver.DevicePtr,
	workspaceBytes uint64,
	stream driver.Stream,
) error {
	alpha, beta := gemmProductScale, gemmAccumulatorScale
	status, _, _ := l.lt.matmul.Call(
		uintptr(handle), plan.desc, uintptr(unsafe.Pointer(&alpha)),
		uintptr(a), plan.a, uintptr(b), plan.b, uintptr(unsafe.Pointer(&beta)),
		uintptr(d), plan.d, uintptr(d), plan.d,
		uintptr(unsafe.Pointer(&plan.algo)), uintptr(workspace), uintptr(workspaceBytes), uintptr(stream),
	)
	return result("cublasLtMatmul", status)
}

// DestroyFP8Plan releases a plan's descriptors.
func (l *Library) DestroyFP8Plan(plan *FP8Plan) error {
	if plan == nil {
		return nil
	}
	var errs []error
	for _, layout := range []uintptr{plan.a, plan.b, plan.d} {
		if layout != 0 {
			status, _, _ := l.lt.layoutDestroy.Call(layout)
			errs = append(errs, result("cublasLtMatrixLayoutDestroy", status))
		}
	}
	if plan.desc != 0 {
		status, _, _ := l.lt.descDestroy.Call(plan.desc)
		errs = append(errs, result("cublasLtMatmulDescDestroy", status))
	}
	*plan = FP8Plan{}
	return errors.Join(errs...)
}

// setAttribute calls a cuBLASLt Set*Attribute entry with value's bytes.
func setAttribute[T any](proc *syscall.Proc, operation string, object, attribute uintptr, value T) error {
	status, _, _ := proc.Call(object, attribute, uintptr(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return result(operation, status)
}
