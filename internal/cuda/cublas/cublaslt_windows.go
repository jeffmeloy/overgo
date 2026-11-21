//go:build windows

package cublas

import (
	"errors"
	"syscall"
	"unsafe"

	"overgo/internal/cuda/driver"
)

// cuBLASLt enumerants follow cublasLt.h from CUDA 12.9.
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
	// CUDA 12.9 cublasLt.h defines bias epilogues and pointer attributes.
	ltDescEpilogue    = uintptr(7)
	ltDescBiasPointer = uintptr(8)
	ltEpilogueBias    = int32(4)
	// ltBF16Alignment admits operands aligned to one 16-byte vector.
	ltBF16Alignment = uint32(16)
	// CUDA 12.9 cublasLt.h assigns these A/B/C/D alignment preference attributes.
	ltAlignmentA = uintptr(5)
	ltAlignmentB = uintptr(6)
	ltAlignmentC = uintptr(7)
	ltAlignmentD = uintptr(8)
)

// ltOperandAlignments names the A, B, C and D minimum-alignment preferences in cublasLt.h.
var ltOperandAlignments = [...]uintptr{ltAlignmentA, ltAlignmentB, ltAlignmentC, ltAlignmentD}

// ltHeuristicResult mirrors cublasLtMatmulHeuristicResult_t.
type ltHeuristicResult struct {
	algo [ltAlgoWords]uint64
	_    [ltHeuristicTailBytes]byte
}

// ltProcs holds the cuBLASLt entry points used by native tensor-core products.
type ltProcs struct {
	create, destroy                                             *syscall.Proc
	descCreate, descDestroy, descSetAttribute                   *syscall.Proc
	layoutCreate, layoutDestroy                                 *syscall.Proc
	preferenceCreate, preferenceDestroy, preferenceSetAttribute *syscall.Proc
	heuristic, matmul                                           *syscall.Proc
}

func (p *ltProcs) bindings() []Binding {
	return []Binding{
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
// MatmulPlan) and takes cuBLASLt's first heuristic algorithm needing at
// most workspaceLimit bytes of workspace. plan owns every descriptor
// created until the product is built: an error return overwrites the
// result, never plan, so a failed construction destroys each one.
func (l *Library) PlanFP8(handle LtHandle, m, n, k, workspaceLimit uint64) (built *MatmulPlan, err error) {
	return l.planMatmul(handle, m, n, k, workspaceLimit, dataF8E4M3, DataF32, false)
}

// PlanBF16Bias selects a BF16 product with native bias and BF16 output rounding.
func (l *Library) PlanBF16Bias(handle LtHandle, m, n, k, workspaceLimit uint64) (*MatmulPlan, error) {
	return l.planMatmul(handle, m, n, k, workspaceLimit, DataBF16, DataBF16, true)
}

func (l *Library) planMatmul(handle LtHandle, m, n, k, workspaceLimit uint64, inputType, outputType DataType, bias bool) (built *MatmulPlan, err error) {
	plan := &MatmulPlan{}
	defer func() {
		if err != nil {
			err = errors.Join(err, l.DestroyMatmulPlan(plan))
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
		{&plan.a, inputType, k, m},
		{&plan.b, inputType, k, n},
		{&plan.d, outputType, m, n},
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
	if bias {
		if err := setAttribute(l.lt.descSetAttribute, "cublasLtMatmulDescSetAttribute", plan.desc, ltDescEpilogue, ltEpilogueBias); err != nil {
			return nil, err
		}
		for _, attribute := range ltOperandAlignments {
			if err := setAttribute(l.lt.preferenceSetAttribute, "cublasLtMatmulPreferenceSetAttribute", preference, attribute, ltBF16Alignment); err != nil {
				return nil, err
			}
		}
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

// Matmul enqueues the planned product with the current bias address.
// The workspace must meet the limit used for algorithm selection.
func (l *Library) Matmul(
	handle LtHandle,
	plan *MatmulPlan,
	a, b, d, bias, workspace driver.DevicePtr,
	workspaceBytes uint64,
	stream driver.Stream,
) error {
	if bias != 0 {
		if err := setAttribute(l.lt.descSetAttribute, "cublasLtMatmulDescSetAttribute", plan.desc, ltDescBiasPointer, bias); err != nil {
			return err
		}
	}
	alpha, beta := gemmProductScale, gemmAccumulatorScale
	status, _, _ := l.lt.matmul.Call(
		uintptr(handle), plan.desc, uintptr(unsafe.Pointer(&alpha)),
		uintptr(a), plan.a, uintptr(b), plan.b, uintptr(unsafe.Pointer(&beta)),
		uintptr(d), plan.d, uintptr(d), plan.d,
		uintptr(unsafe.Pointer(&plan.algo)), uintptr(workspace), uintptr(workspaceBytes), uintptr(stream),
	)
	return result("cublasLtMatmul", status)
}

// DestroyMatmulPlan releases a plan's descriptors.
func (l *Library) DestroyMatmulPlan(plan *MatmulPlan) error {
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
	*plan = MatmulPlan{}
	return errors.Join(errs...)
}

// setAttribute calls a cuBLASLt Set*Attribute entry with value's bytes.
func setAttribute[T any](proc *syscall.Proc, operation string, object, attribute uintptr, value T) error {
	status, _, _ := proc.Call(object, attribute, uintptr(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return result(operation, status)
}
