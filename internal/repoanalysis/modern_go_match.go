package repoanalysis

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"path"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

var modernGoAdoptedCalls = map[string][]string{
	"new_expression": {"new"}, "errors_as_type": {"errors.AsType"},
	"sync_waitgroup_go": {"WaitGroup.Go"}, "testing_t_context": {"T.Context"},
	"testing_b_loop": {"B.Loop"}, "strings_split_seq": {"strings.SplitSeq", "strings.FieldsSeq", "bytes.SplitSeq", "bytes.FieldsSeq"},
	"maps_keys_values_iter": {"maps.Keys", "maps.Values"}, "slices_collect": {"slices.Collect"},
	"slices_sorted": {"slices.Sorted", "slices.SortedFunc", "slices.SortedStableFunc"}, "time_tick_gc": {"time.Tick"},
	"cmp_or": {"cmp.Or"}, "reflect_type_for": {"reflect.TypeFor"}, "min_max": {"min", "max"},
	"clear": {"clear"}, "slices_contains": {"slices.Contains", "slices.ContainsFunc"},
	"slices_index": {"slices.Index"}, "slices_index_func": {"slices.IndexFunc"},
	"slices_sort_func": {"slices.SortFunc", "slices.SortStableFunc"}, "slices_sort": {"slices.Sort"},
	"slices_max_min": {"slices.Max", "slices.Min", "slices.MaxFunc", "slices.MinFunc"},
	"slices_reverse": {"slices.Reverse"}, "slices_compact": {"slices.Compact", "slices.CompactFunc"},
	"slices_clip": {"slices.Clip"}, "slices_clone": {"slices.Clone"}, "maps_clone": {"maps.Clone"},
	"maps_copy": {"maps.Copy", "maps.Insert"}, "maps_delete_func": {"maps.DeleteFunc"},
	"sync_once_func": {"sync.OnceFunc"}, "sync_once_value": {"sync.OnceValue", "sync.OnceValues"},
	"context_after_func": {"context.AfterFunc"}, "context_timeout_deadline_cause": {"context.WithTimeoutCause", "context.WithDeadlineCause"},
	"strings_clone": {"strings.Clone"}, "bytes_clone": {"bytes.Clone"},
	"strings_cut_prefix_suffix": {"strings.CutPrefix", "strings.CutSuffix"}, "errors_join": {"errors.Join"},
	"context_cancel_cause": {"context.WithCancelCause", "context.Cause"}, "fmt_appendf": {"fmt.Appendf", "fmt.Append"},
	"atomic_types": {"atomic.Bool", "atomic.Int32", "atomic.Int64", "atomic.Uint32", "atomic.Uint64", "atomic.Uintptr", "atomic.Pointer"},
	"any":          {"any"}, "bytes_cut": {"bytes.Cut"}, "strings_cut": {"strings.Cut"},
	"errors_is": {"errors.Is"}, "time_until": {"time.Until"}, "time_since": {"time.Since"},
}

func modernGoNodeMatch(id string, node ast.Node, info *types.Info, imports map[string]string, testFile bool) (bool, bool) {
	call, isCall := node.(*ast.CallExpr)
	key := ""
	if isCall {
		key = modernGoCallKey(call, info, imports)
	}
	adopted := containsModernGoCall(modernGoAdoptedCalls[id], key)
	switch id {
	case "new_expression":
		return modernGoNewExpressionCandidate(node), adopted
	case "errors_as_type":
		return key == "errors.As", adopted
	case "sync_waitgroup_go":
		block, ok := node.(*ast.BlockStmt)
		return ok && modernGoWaitGroupLaunchBlock(block, info, imports), adopted
	case "testing_t_context":
		return testFile && (key == "context.Background" || key == "context.TODO") &&
			modernGoTestingOwnerAt(call.Pos(), info) != "", adopted
	case "json_omitzero":
		field, ok := node.(*ast.Field)
		return ok && modernGoOmitEmptyCandidate(field, info), modernGoStructTag(field, "omitzero")
	case "testing_b_loop":
		return modernGoBenchmarkLoopCandidate(node, info), adopted
	case "strings_split_seq":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && callExpressionKey(rangeStatement.X, info, imports, "strings.Split", "strings.Fields", "bytes.Split", "bytes.Fields"), adopted
	case "maps_keys_values_iter":
		rangeStatement, ok := node.(*ast.RangeStmt)
		_, _, projection := modernGoMapProjectionLoop(rangeStatement, info)
		return ok && projection, adopted
	case "slices_collect":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && modernGoIteratorCollectLoop(rangeStatement, info), adopted
	case "slices_sorted":
		block, ok := node.(*ast.BlockStmt)
		return ok && modernGoSortedMapProjection(block, info, imports), adopted
	case "time_tick_gc":
		return key == "time.NewTicker", adopted
	case "range_over_int":
		loop, ok := node.(*ast.ForStmt)
		return ok && modernGoCountingLoop(loop, info), adopted
	case "loopvar_capture":
		assignment, ok := node.(*ast.AssignStmt)
		return ok && modernGoSelfAssignment(assignment) || modernGoForwardedLoopCapture(node), adopted
	case "cmp_or":
		statement, ok := node.(*ast.IfStmt)
		return ok && modernGoFallbackIf(statement), adopted
	case "reflect_type_for":
		return key == "reflect.TypeOf" && isReflectTypeToken(call), adopted
	case "http_servemux_patterns":
		return modernGoServeMuxPattern(call, key)
	case "min_max":
		statement, ok := node.(*ast.IfStmt)
		return key == "math.Min" || key == "math.Max" || ok && modernGoComparisonAssignment(statement, info), adopted
	case "clear":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && modernGoClearLoop(rangeStatement, info), adopted
	case "slices_contains", "slices_index", "slices_index_func":
		loop, ok := node.(*ast.RangeStmt)
		return ok && modernGoRangeType(loop, info, "slice") && modernGoSliceSearch(loop, id), adopted
	case "slices_sort":
		return key == "sort.Strings" || key == "sort.Ints" || key == "sort.Float64s", adopted
	case "slices_sort_func":
		return key == "sort.Slice" || key == "sort.SliceStable", adopted
	case "slices_max_min":
		loop, ok := node.(*ast.RangeStmt)
		return ok && modernGoRangeType(loop, info, "slice") && modernGoSliceMinMax(loop, info), adopted
	case "slices_reverse":
		loop, ok := node.(*ast.ForStmt)
		return ok && modernGoSliceReverseLoop(loop, info), adopted
	case "slices_compact":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && modernGoRangeType(rangeStatement, info, "slice") && modernGoSliceCompact(rangeStatement), adopted
	case "slices_clip":
		slice, ok := node.(*ast.SliceExpr)
		return ok && slice.Slice3 && modernGoEquivalentExpression(slice.High, slice.Max), adopted
	case "slices_clone":
		return isCall && key == "append" && len(call.Args) > 1 && isNilSlice(call.Args[0]), adopted
	case "maps_clone":
		block, ok := node.(*ast.BlockStmt)
		return ok && modernGoMapCloneBlock(block, info), adopted
	case "maps_copy":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && modernGoMapCopyLoop(rangeStatement, info), adopted
	case "maps_delete_func":
		rangeStatement, ok := node.(*ast.RangeStmt)
		return ok && modernGoMapDeleteLoop(rangeStatement, info, true), adopted
	case "sync_once_func":
		return key == "Once.Do" && !modernGoOnceValueCall(call, info), adopted
	case "sync_once_value":
		return key == "Once.Do" && modernGoOnceValueCall(call, info), adopted
	case "context_after_func":
		statement, ok := node.(*ast.GoStmt)
		return ok && modernGoContextCleanupGoroutine(statement, info), adopted
	case "context_timeout_deadline_cause":
		return key == "context.WithTimeout" || key == "context.WithDeadline", adopted
	case "strings_clone":
		return isCall && key == "string" && len(call.Args) == 1 && callExpressionKey(call.Args[0], info, imports, "[]byte"), adopted
	case "bytes_clone":
		return isCall && key == "append" && modernGoByteCloneAppend(call, info), adopted
	case "strings_cut_prefix_suffix":
		statement, ok := node.(*ast.IfStmt)
		return ok && modernGoCutPrefixSuffixIf(statement, info, imports), adopted
	case "errors_join":
		return key == "fmt.Errorf" && modernGoMultipleWrappedErrors(call), adopted
	case "context_cancel_cause":
		return key == "context.WithCancel", adopted
	case "fmt_appendf":
		return isCall && key == "append" && modernGoAppendFormat(call, info, imports), adopted
	case "atomic_types":
		return strings.HasPrefix(key, "atomic.Load") || strings.HasPrefix(key, "atomic.Store") ||
			strings.HasPrefix(key, "atomic.Add") || strings.HasPrefix(key, "atomic.Swap") ||
			strings.HasPrefix(key, "atomic.CompareAndSwap"), adopted
	case "any":
		interfaceType, ok := node.(*ast.InterfaceType)
		return ok && interfaceType.Methods != nil && len(interfaceType.Methods.List) == 0, adopted
	case "bytes_cut":
		statement, ok := node.(*ast.IfStmt)
		return ok && modernGoIndexCutIf(statement, info, imports, "bytes.Index", "bytes.IndexByte"), adopted
	case "strings_cut":
		statement, ok := node.(*ast.IfStmt)
		return ok && modernGoIndexCutIf(statement, info, imports, "strings.Index", "strings.IndexByte"), adopted
	case "errors_is":
		binary, ok := node.(*ast.BinaryExpr)
		comparison := ok && (binary.Op.String() == "==" || binary.Op.String() == "!=")
		return comparison && expressionIsError(binary.X, info) && expressionIsError(binary.Y, info), adopted
	case "time_until":
		return isCall && strings.HasSuffix(key, ".Sub") && astContainsQualifiedCall(call, info, imports, "time.Now"), adopted
	case "time_since":
		return key == "Time.Sub" && selectorReceiverCall(call, info, imports, "time.Now"), adopted
	}
	return false, adopted
}

func modernGoTestingOwnerAt(position token.Pos, info *types.Info) string {
	if info == nil || !position.IsValid() {
		return ""
	}
	var scope *types.Scope
	for _, candidate := range info.Scopes {
		if candidate.Contains(position) && (scope == nil || candidate.Pos() >= scope.Pos() && candidate.End() <= scope.End()) {
			scope = candidate
		}
	}
	for current := scope; current != nil; current = current.Parent() {
		for _, name := range current.Names() {
			object, ok := current.Lookup(name).(*types.Var)
			if !ok || !modernGoTestingOwnerType(object.Type()) {
				continue
			}
			return name
		}
	}
	return ""
}

func modernGoTestingOwnerType(value types.Type) bool {
	pointer, ok := value.(*types.Pointer)
	if !ok {
		return false
	}
	named, ok := pointer.Elem().(*types.Named)
	if !ok || named.Obj().Pkg() == nil || named.Obj().Pkg().Path() != "testing" {
		return false
	}
	return named.Obj().Name() == "T" || named.Obj().Name() == "B"
}

func modernGoContextCleanupGoroutine(statement *ast.GoStmt, info *types.Info) bool {
	if statement == nil || statement.Call == nil {
		return false
	}
	function, ok := statement.Call.Fun.(*ast.FuncLit)
	if !ok || len(function.Body.List) == 0 {
		return false
	}
	return modernGoContextDoneWait(function.Body.List, info)
}

func modernGoContextDoneWait(statements []ast.Stmt, info *types.Info) bool {
	if len(statements) == 0 {
		return false
	}
	switch current := statements[0].(type) {
	case *ast.ExprStmt:
		receive, ok := current.X.(*ast.UnaryExpr)
		return len(statements) > 1 && ok && receive.Op == token.ARROW && astContainsContextDoneCall(receive.X, info)
	case *ast.SelectStmt:
		if current.Body == nil || len(current.Body.List) != 1 {
			return false
		}
		clause, ok := current.Body.List[0].(*ast.CommClause)
		if !ok || clause.Comm == nil || len(clause.Body) == 0 {
			return false
		}
		expression, ok := clause.Comm.(*ast.ExprStmt)
		if !ok {
			return false
		}
		receive, ok := expression.X.(*ast.UnaryExpr)
		return ok && receive.Op == token.ARROW && astContainsContextDoneCall(receive.X, info)
	}
	return false
}

func astContainsContextDoneCall(node ast.Node, info *types.Info) bool {
	for current := range ast.Preorder(node) {
		call, ok := current.(*ast.CallExpr)
		if !ok {
			continue
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "Done" {
			continue
		}
		selection := info.Selections[selector]
		if selection != nil && selection.Obj().Pkg() != nil && selection.Obj().Pkg().Path() == "context" {
			return true
		}
	}
	return false
}

func modernGoNewExpressionCandidate(node ast.Node) bool {
	switch value := node.(type) {
	case *ast.FuncDecl:
		return modernGoPointerHelper(value.Type.Params, value.Body)
	case *ast.FuncLit:
		return modernGoPointerHelper(value.Type.Params, value.Body)
	case *ast.BlockStmt:
		return modernGoTemporaryAddress(value)
	default:
		return false
	}
}

func modernGoPointerHelper(parameters *ast.FieldList, body *ast.BlockStmt) bool {
	if parameters == nil || body == nil || len(body.List) == 0 {
		return false
	}
	parameterNames := map[string]bool{}
	for _, field := range parameters.List {
		for _, name := range field.Names {
			parameterNames[name.Name] = true
		}
	}
	return modernGoReturnedAddress(body.List[len(body.List)-1], parameterNames)
}

func modernGoTemporaryAddress(block *ast.BlockStmt) bool {
	for index, statement := range block.List {
		if index == 0 {
			continue
		}
		assigned := modernGoAssignedNames(block.List[index-1])
		if modernGoReturnedAddress(statement, assigned) {
			return true
		}
	}
	return false
}

func modernGoAssignedNames(statement ast.Stmt) map[string]bool {
	result := map[string]bool{}
	switch value := statement.(type) {
	case *ast.AssignStmt:
		for _, expression := range value.Lhs {
			if name, ok := expression.(*ast.Ident); ok {
				result[name.Name] = true
			}
		}
	case *ast.DeclStmt:
		declaration, ok := value.Decl.(*ast.GenDecl)
		if !ok {
			return result
		}
		for _, specification := range declaration.Specs {
			if values, ok := specification.(*ast.ValueSpec); ok {
				for _, name := range values.Names {
					result[name.Name] = true
				}
			}
		}
	}
	return result
}

func modernGoReturnedAddress(statement ast.Stmt, names map[string]bool) bool {
	returned, ok := statement.(*ast.ReturnStmt)
	if !ok || len(returned.Results) != 1 {
		return false
	}
	unary, ok := returned.Results[0].(*ast.UnaryExpr)
	if !ok || unary.Op.String() != "&" {
		return false
	}
	name, ok := unary.X.(*ast.Ident)
	return ok && names[name.Name]
}

func modernGoCallKey(call *ast.CallExpr, info *types.Info, imports map[string]string) string {
	if call == nil {
		return ""
	}
	switch function := call.Fun.(type) {
	case *ast.Ident:
		return function.Name
	case *ast.ArrayType:
		return "[]byte"
	case *ast.SelectorExpr:
		if base, ok := function.X.(*ast.Ident); ok {
			if importPath := imports[base.Name]; importPath != "" {
				return path.Base(importPath) + "." + function.Sel.Name
			}
		}
		if info != nil {
			if selection := info.Selections[function]; selection != nil {
				receiver := selection.Recv()
				if pointer, ok := receiver.(*types.Pointer); ok {
					receiver = pointer.Elem()
				}
				if named, ok := receiver.(*types.Named); ok {
					return named.Obj().Name() + "." + function.Sel.Name
				}
			}
		}
		return function.Sel.Name
	}
	return ""
}

func containsModernGoCall(values []string, value string) bool {
	return slices.Contains(values, value)
}

func modernGoRangeType(statement *ast.RangeStmt, info *types.Info, want string) bool {
	if statement == nil || info == nil {
		return false
	}
	typeOf := info.TypeOf(statement.X)
	if typeOf == nil {
		return false
	}
	switch typeOf.Underlying().(type) {
	case *types.Map:
		return want == "map"
	case *types.Slice, *types.Array:
		return want == "slice"
	case *types.Signature:
		return want == "iterator"
	}
	return false
}

func modernGoMapCopyLoop(statement *ast.RangeStmt, info *types.Info) bool {
	if !modernGoRangeType(statement, info, "map") || len(statement.Body.List) != 1 {
		return false
	}
	key, keyOK := statement.Key.(*ast.Ident)
	value, valueOK := statement.Value.(*ast.Ident)
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || !keyOK || !valueOK || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return false
	}
	destination, ok := assignment.Lhs[0].(*ast.IndexExpr)
	return ok && directIdent(destination.Index, key.Name) && directIdent(assignment.Rhs[0], value.Name)
}

func modernGoOnceValueCall(call *ast.CallExpr, info *types.Info) bool {
	if call == nil || info == nil || len(call.Args) != 1 {
		return false
	}
	function, ok := call.Args[0].(*ast.FuncLit)
	if !ok {
		return false
	}
	writesOuter := false
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if writesOuter {
			return false
		}
		switch current := node.(type) {
		case *ast.AssignStmt:
			for _, expression := range current.Lhs {
				if modernGoWritesOuterExpression(expression, function.Pos(), info) {
					writesOuter = true
					break
				}
			}
		case *ast.IncDecStmt:
			writesOuter = modernGoWritesOuterExpression(current.X, function.Pos(), info)
		case *ast.FuncLit:
			return false
		}
		return !writesOuter
	})
	return writesOuter
}

func modernGoWritesOuterExpression(expression ast.Expr, functionPosition token.Pos, info *types.Info) bool {
	switch current := expression.(type) {
	case *ast.Ident:
		object := info.ObjectOf(current)
		return object != nil && object.Pos().IsValid() && object.Pos() < functionPosition
	case *ast.SelectorExpr:
		return modernGoWritesOuterExpression(current.X, functionPosition, info)
	case *ast.IndexExpr:
		return modernGoWritesOuterExpression(current.X, functionPosition, info)
	case *ast.StarExpr:
		return modernGoWritesOuterExpression(current.X, functionPosition, info)
	case *ast.ParenExpr:
		return modernGoWritesOuterExpression(current.X, functionPosition, info)
	}
	return false
}

func modernGoWaitGroupLaunchBlock(block *ast.BlockStmt, info *types.Info, imports map[string]string) bool {
	if block == nil {
		return false
	}
	for index, statement := range block.List {
		following := block.List[index+1:]
		if len(following) == 0 {
			break
		}
		addStatement, ok := statement.(*ast.ExprStmt)
		if !ok {
			continue
		}
		add, ok := addStatement.X.(*ast.CallExpr)
		if !ok || modernGoCallKey(add, info, imports) != "WaitGroup.Add" || len(add.Args) != 1 || !basicLiteral(add.Args[0], "1") {
			continue
		}
		addSelector, ok := add.Fun.(*ast.SelectorExpr)
		launch, launchOK := following[0].(*ast.GoStmt)
		if !ok || !launchOK || len(launch.Call.Args) != 0 {
			continue
		}
		function, ok := launch.Call.Fun.(*ast.FuncLit)
		if !ok || function.Body == nil || len(function.Body.List) == 0 || astContainsCall(function.Body, "panic") {
			continue
		}
		deferred, ok := function.Body.List[0].(*ast.DeferStmt)
		if !ok || modernGoCallKey(deferred.Call, info, imports) != "WaitGroup.Done" {
			continue
		}
		doneSelector, ok := deferred.Call.Fun.(*ast.SelectorExpr)
		if ok && expressionsEqual(addSelector.X, doneSelector.X) {
			return true
		}
	}
	return false
}

func modernGoByteCloneAppend(call *ast.CallExpr, info *types.Info) bool {
	if call == nil || info == nil || !call.Ellipsis.IsValid() || !modernGoBinaryExpressions(call.Args) || !isNilSlice(call.Args[0]) {
		return false
	}
	return modernGoByteSliceType(info.TypeOf(call))
}

func modernGoAppendFormat(call *ast.CallExpr, info *types.Info, imports map[string]string) bool {
	if call == nil || info == nil || !call.Ellipsis.IsValid() || !modernGoBinaryExpressions(call.Args) ||
		!modernGoByteSliceType(info.TypeOf(call)) {
		return false
	}
	formatted, ok := call.Args[1].(*ast.CallExpr)
	return ok && modernGoCallKey(formatted, info, imports) == "fmt.Sprintf"
}

func modernGoByteSliceType(typeOf types.Type) bool {
	if typeOf == nil {
		return false
	}
	slice, ok := typeOf.Underlying().(*types.Slice)
	if !ok {
		return false
	}
	element, ok := slice.Elem().Underlying().(*types.Basic)
	return ok && element.Kind() == types.Uint8
}

func modernGoCutPrefixSuffixIf(statement *ast.IfStmt, info *types.Info, imports map[string]string) bool {
	if statement == nil || statement.Init != nil || statement.Cond == nil {
		return false
	}
	condition, ok := statement.Cond.(*ast.CallExpr)
	if !ok || !modernGoBinaryExpressions(condition.Args) {
		return false
	}
	want := ""
	switch modernGoCallKey(condition, info, imports) {
	case "strings.HasPrefix":
		want = "strings.TrimPrefix"
	case "strings.HasSuffix":
		want = "strings.TrimSuffix"
	default:
		return false
	}
	found := false
	ast.Inspect(statement.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || modernGoCallKey(call, info, imports) != want || len(call.Args) != len(condition.Args) {
			return !found
		}
		found = slices.EqualFunc(call.Args, condition.Args, modernGoEquivalentExpression)
		return !found
	})
	return found
}

func modernGoIndexCutIf(statement *ast.IfStmt, info *types.Info, imports map[string]string, calls ...string) bool {
	if statement == nil || statement.Init == nil || statement.Cond == nil {
		return false
	}
	initial, ok := statement.Init.(*ast.AssignStmt)
	if !ok || len(initial.Lhs) != 1 || len(initial.Rhs) != 1 {
		return false
	}
	index, ok := initial.Lhs[0].(*ast.Ident)
	call, callOK := initial.Rhs[0].(*ast.CallExpr)
	if !ok || !callOK || len(call.Args) == 0 || !slices.Contains(calls, modernGoCallKey(call, info, imports)) ||
		!modernGoIndexFoundCondition(statement.Cond, index.Name) {
		return false
	}
	found := false
	ast.Inspect(statement.Body, func(node ast.Node) bool {
		slice, ok := node.(*ast.SliceExpr)
		if !ok || !modernGoEquivalentExpression(slice.X, call.Args[0]) {
			return !found
		}
		found = astContainsIdent(slice.Low, index.Name) || astContainsIdent(slice.High, index.Name)
		return !found
	})
	return found
}

func modernGoIndexFoundCondition(expression ast.Expr, index string) bool {
	binary, ok := expression.(*ast.BinaryExpr)
	if !ok || !directIdent(binary.X, index) {
		return false
	}
	switch binary.Op {
	case token.GEQ:
		return basicLiteral(binary.Y, "0")
	case token.NEQ:
		minusOne, ok := binary.Y.(*ast.UnaryExpr)
		return ok && minusOne.Op == token.SUB && basicLiteral(minusOne.X, "1")
	}
	return false
}

func modernGoMapProjectionLoop(statement *ast.RangeStmt, info *types.Info) (string, string, bool) {
	if !modernGoRangeType(statement, info, "map") || len(statement.Body.List) != 1 {
		return "", "", false
	}
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return "", "", false
	}
	destination, ok := assignment.Lhs[0].(*ast.Ident)
	if !ok {
		return "", "", false
	}
	appendCall, ok := assignment.Rhs[0].(*ast.CallExpr)
	if !ok || appendCall.Ellipsis.IsValid() || !modernGoBinaryExpressions(appendCall.Args) ||
		!directIdent(appendCall.Fun, "append") || !directIdent(appendCall.Args[0], destination.Name) {
		return "", "", false
	}
	appended, ok := appendCall.Args[1].(*ast.Ident)
	if !ok || appended.Name == "_" {
		return "", "", false
	}
	if key, ok := statement.Key.(*ast.Ident); ok && key.Name == appended.Name {
		return destination.Name, "keys", true
	}
	if value, ok := statement.Value.(*ast.Ident); ok && value.Name == appended.Name {
		return destination.Name, "values", true
	}
	return "", "", false
}

func modernGoIteratorCollectLoop(statement *ast.RangeStmt, info *types.Info) bool {
	if !modernGoRangeType(statement, info, "iterator") {
		return false
	}
	_, _, ok := modernGoAppendRangeLoop(statement)
	return ok
}

func modernGoAppendRangeLoop(statement *ast.RangeStmt) (string, string, bool) {
	if statement == nil || len(statement.Body.List) != 1 {
		return "", "", false
	}
	value, ok := statement.Value.(*ast.Ident)
	if !ok || value.Name == "_" {
		value, ok = statement.Key.(*ast.Ident)
	}
	if !ok || value.Name == "_" {
		return "", "", false
	}
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return "", "", false
	}
	destination, ok := assignment.Lhs[0].(*ast.Ident)
	if !ok {
		return "", "", false
	}
	appendCall, ok := assignment.Rhs[0].(*ast.CallExpr)
	if !ok || appendCall.Ellipsis.IsValid() || !modernGoBinaryExpressions(appendCall.Args) ||
		!directIdent(appendCall.Fun, "append") || !directIdent(appendCall.Args[0], destination.Name) ||
		!directIdent(appendCall.Args[1], value.Name) {
		return "", "", false
	}
	return destination.Name, value.Name, true
}

func modernGoSortedMapProjection(block *ast.BlockStmt, info *types.Info, imports map[string]string) bool {
	if block == nil {
		return false
	}
	for index, statement := range block.List {
		loop, ok := statement.(*ast.RangeStmt)
		if !ok {
			continue
		}
		destination, _, ok := modernGoMapProjectionLoop(loop, info)
		if !ok || len(block.List[index+1:]) == 0 {
			continue
		}
		expression, ok := block.List[index+1].(*ast.ExprStmt)
		if !ok {
			continue
		}
		call, ok := expression.X.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 || !directIdent(call.Args[0], destination) {
			continue
		}
		switch modernGoCallKey(call, info, imports) {
		case "sort.Strings", "sort.Ints", "sort.Float64s", "slices.Sort":
			return true
		}
	}
	return false
}

func modernGoMapCloneBlock(block *ast.BlockStmt, info *types.Info) bool {
	if block == nil {
		return false
	}
	for index, statement := range block.List {
		following := block.List[index+1:]
		if len(following) <= 1 {
			break
		}
		initial, ok := statement.(*ast.AssignStmt)
		if !ok || initial.Tok != token.DEFINE || len(initial.Lhs) != 1 || len(initial.Rhs) != 1 {
			continue
		}
		destination, ok := initial.Lhs[0].(*ast.Ident)
		makeCall, makeOK := initial.Rhs[0].(*ast.CallExpr)
		if !ok || !makeOK {
			continue
		}
		makeName, nameOK := makeCall.Fun.(*ast.Ident)
		if !nameOK || makeName.Name != "make" || len(makeCall.Args) == 0 {
			continue
		}
		if _, mapType := makeCall.Args[0].(*ast.MapType); !mapType {
			continue
		}
		loop, loopOK := following[0].(*ast.RangeStmt)
		returned, returnOK := following[1].(*ast.ReturnStmt)
		if !loopOK || !returnOK || len(returned.Results) != 1 || !directIdent(returned.Results[0], destination.Name) ||
			!modernGoMapCopyLoop(loop, info) {
			continue
		}
		assignment := loop.Body.List[0].(*ast.AssignStmt)
		indexed := assignment.Lhs[0].(*ast.IndexExpr)
		if directIdent(indexed.X, destination.Name) {
			return true
		}
	}
	return false
}

func modernGoCountingLoop(loop *ast.ForStmt, info *types.Info) bool {
	if loop == nil || loop.Init == nil || loop.Cond == nil || loop.Post == nil {
		return false
	}
	initial, ok := loop.Init.(*ast.AssignStmt)
	if !ok || initial.Tok != token.DEFINE || len(initial.Lhs) != 1 || len(initial.Rhs) != 1 {
		return false
	}
	zero, ok := initial.Rhs[0].(*ast.BasicLit)
	condition, conditionOK := loop.Cond.(*ast.BinaryExpr)
	increment, incrementOK := loop.Post.(*ast.IncDecStmt)
	if !ok || zero.Value != "0" || !conditionOK || condition.Op.String() != "<" || !incrementOK || increment.Tok.String() != "++" {
		return false
	}
	counter, counterOK := initial.Lhs[0].(*ast.Ident)
	conditionCounter, conditionCounterOK := condition.X.(*ast.Ident)
	incrementCounter, incrementCounterOK := increment.X.(*ast.Ident)
	if !counterOK || !conditionCounterOK || !incrementCounterOK ||
		counter.Name != conditionCounter.Name || counter.Name != incrementCounter.Name {
		return false
	}
	if info != nil && !modernGoIntegerType(info.TypeOf(counter)) {
		return false
	}
	if !modernGoStableRangeBound(condition.Y) {
		return false
	}
	protected := map[string]bool{counter.Name: true}
	ast.Inspect(condition.Y, func(node ast.Node) bool {
		if identifier, ok := node.(*ast.Ident); ok {
			protected[identifier.Name] = true
		}
		return true
	})
	return !astMutatesIdentifiers(loop.Body, protected)
}

func modernGoIntegerType(value types.Type) bool {
	if value == nil {
		return false
	}
	basic, ok := value.Underlying().(*types.Basic)
	return ok && basic.Info()&types.IsInteger != 0
}

func modernGoStableRangeBound(expression ast.Expr) bool {
	switch value := expression.(type) {
	case *ast.BasicLit, *ast.Ident:
		return true
	case *ast.ParenExpr:
		return modernGoStableRangeBound(value.X)
	case *ast.UnaryExpr:
		return modernGoStableRangeBound(value.X)
	case *ast.BinaryExpr:
		return modernGoStableRangeBound(value.X) && modernGoStableRangeBound(value.Y)
	case *ast.CallExpr:
		function, ok := value.Fun.(*ast.Ident)
		return ok && (function.Name == "len" || function.Name == "cap") &&
			slices.ContainsFunc(value.Args, modernGoStableRangeBound)
	default:
		return false
	}
}

func astMutatesIdentifiers(root ast.Node, names map[string]bool) bool {
	for node := range ast.Preorder(root) {
		var expressions []ast.Expr
		switch statement := node.(type) {
		case *ast.AssignStmt:
			expressions = statement.Lhs
		case *ast.IncDecStmt:
			expressions = []ast.Expr{statement.X}
		case *ast.RangeStmt:
			expressions = []ast.Expr{statement.Key, statement.Value}
		}
		for _, expression := range expressions {
			if identifier, ok := expression.(*ast.Ident); ok && names[identifier.Name] {
				return true
			}
		}
	}
	return false
}

func modernGoSelfAssignment(assignment *ast.AssignStmt) bool {
	if assignment == nil || assignment.Tok.String() != ":=" || len(assignment.Lhs) != len(assignment.Rhs) {
		return false
	}
	for index := range assignment.Lhs {
		left, leftOK := assignment.Lhs[index].(*ast.Ident)
		right, rightOK := assignment.Rhs[index].(*ast.Ident)
		if leftOK && rightOK && left.Name == right.Name {
			return true
		}
	}
	return false
}

func modernGoForwardedLoopCapture(node ast.Node) bool {
	captured := map[string]bool{}
	var body *ast.BlockStmt
	switch loop := node.(type) {
	case *ast.RangeStmt:
		body = loop.Body
		for _, expression := range []ast.Expr{loop.Key, loop.Value} {
			if identifier, ok := expression.(*ast.Ident); ok && identifier.Name != "_" {
				captured[identifier.Name] = true
			}
		}
	case *ast.ForStmt:
		body = loop.Body
		if initial, ok := loop.Init.(*ast.AssignStmt); ok {
			for _, expression := range initial.Lhs {
				if identifier, ok := expression.(*ast.Ident); ok {
					captured[identifier.Name] = true
				}
			}
		}
	default:
		return false
	}
	if body == nil || len(captured) == 0 {
		return false
	}
	found := false
	ast.Inspect(body, func(current ast.Node) bool {
		if found {
			return false
		}
		switch current.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return false
		}
		statement, ok := current.(*ast.GoStmt)
		if !ok {
			return true
		}
		function, ok := statement.Call.Fun.(*ast.FuncLit)
		if !ok {
			return true
		}
		var parameters []string
		for _, field := range function.Type.Params.List {
			for _, name := range field.Names {
				parameters = append(parameters, name.Name)
			}
		}
		if len(parameters) == 0 || len(parameters) != len(statement.Call.Args) {
			return true
		}
		for index, parameter := range parameters {
			argument, ok := statement.Call.Args[index].(*ast.Ident)
			if !ok || argument.Name != parameter || !captured[parameter] {
				return true
			}
		}
		found = true
		return false
	})
	return found
}

func modernGoClearLoop(statement *ast.RangeStmt, info *types.Info) bool {
	if statement == nil || len(statement.Body.List) != 1 {
		return false
	}
	if modernGoRangeType(statement, info, "map") {
		return modernGoMapDeleteLoop(statement, info, false)
	}
	if modernGoRangeType(statement, info, "slice") {
		key, keyOK := statement.Key.(*ast.Ident)
		assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
		if !ok || !keyOK || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 || !isZeroExpression(assignment.Rhs[0]) {
			return false
		}
		indexed, ok := assignment.Lhs[0].(*ast.IndexExpr)
		return ok && directIdent(indexed.X, expressionIdentName(statement.X)) && directIdent(indexed.Index, key.Name)
	}
	return false
}

func modernGoMapDeleteLoop(statement *ast.RangeStmt, info *types.Info, conditional bool) bool {
	if !modernGoRangeType(statement, info, "map") || len(statement.Body.List) != 1 {
		return false
	}
	key, ok := statement.Key.(*ast.Ident)
	if !ok || key.Name == "_" {
		return false
	}
	body := statement.Body.List
	if conditional {
		condition, ok := body[0].(*ast.IfStmt)
		if !ok || condition.Else != nil || len(condition.Body.List) != 1 {
			return false
		}
		body = condition.Body.List
	}
	expression, ok := body[0].(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := expression.X.(*ast.CallExpr)
	if !ok {
		return false
	}
	function, functionOK := call.Fun.(*ast.Ident)
	return functionOK && function.Name == "delete" && len(call.Args) == len([]ast.Expr{statement.X, key}) &&
		expressionsEqual(call.Args[0], statement.X) && directIdent(call.Args[1], key.Name)
}

func expressionIdentName(expression ast.Expr) string {
	identifier, ok := expression.(*ast.Ident)
	if !ok {
		return ""
	}
	return identifier.Name
}

func modernGoOmitEmptyCandidate(field *ast.Field, info *types.Info) bool {
	if !modernGoStructTag(field, "omitempty") || info == nil {
		return false
	}
	typeOf := info.TypeOf(field.Type)
	if typeOf == nil {
		return false
	}
	switch underlying := typeOf.Underlying().(type) {
	case *types.Basic:
		return underlying.Kind() != types.Invalid
	case *types.Struct:
		return true
	}
	return false
}

func modernGoServeMuxPattern(call *ast.CallExpr, key string) (bool, bool) {
	if key != "ServeMux.Handle" && key != "ServeMux.HandleFunc" || call == nil || len(call.Args) == 0 {
		return false, false
	}
	literal, ok := call.Args[0].(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return true, false
	}
	pattern, err := strconv.Unquote(literal.Value)
	if err != nil {
		return true, false
	}
	method, target, methodAware := strings.Cut(pattern, " ")
	methodAware = methodAware && method != "" && target != "" && !strings.ContainsAny(method, "/\t")
	return !methodAware, methodAware
}

func modernGoOmitZeroWireEquivalent(field *ast.Field, info *types.Info) bool {
	if !modernGoStructTag(field, "omitempty") || info == nil {
		return false
	}
	typeOf := info.TypeOf(field.Type)
	if typeOf == nil || modernGoHasIsZeroMethod(typeOf) {
		return false
	}
	basic, ok := typeOf.Underlying().(*types.Basic)
	if !ok {
		return false
	}
	return basic.Info()&(types.IsBoolean|types.IsInteger|types.IsFloat|types.IsString) != 0
}

func modernGoHasIsZeroMethod(typeOf types.Type) bool {
	typesToInspect := []types.Type{typeOf}
	if _, pointer := typeOf.(*types.Pointer); !pointer {
		typesToInspect = append(typesToInspect, types.NewPointer(typeOf))
	}
	for _, candidate := range typesToInspect {
		methods := types.NewMethodSet(candidate)
		for index := range methods.Len() {
			if methods.At(index).Obj().Name() == "IsZero" {
				return true
			}
		}
	}
	return false
}

func modernGoStructTag(field *ast.Field, option string) bool {
	if field == nil || field.Tag == nil {
		return false
	}
	value, err := strconv.Unquote(field.Tag.Value)
	if err != nil {
		return false
	}
	tag := reflect.StructTag(value).Get("json")
	for part := range strings.SplitSeq(tag, ",") {
		if part == option {
			return true
		}
	}
	return false
}

func expressionIsError(expression ast.Expr, info *types.Info) bool {
	if info == nil || info.TypeOf(expression) == nil {
		return false
	}
	errorType := types.Universe.Lookup("error").Type().Underlying().(*types.Interface).Complete()
	// Implements tolerates invalid embedded fields to suppress cascading
	// compiler errors. A rewrite requires an actual matching method instead.
	missing, _ := types.MissingMethod(info.TypeOf(expression), errorType, true)
	return missing == nil
}

func callExpressionKey(expression ast.Expr, info *types.Info, imports map[string]string, keys ...string) bool {
	call, ok := expression.(*ast.CallExpr)
	return ok && containsModernGoCall(keys, modernGoCallKey(call, info, imports))
}

func astContainsCall(node ast.Node, name string) bool {
	if node == nil {
		return false
	}
	for candidate := range ast.Preorder(node) {
		if call, ok := candidate.(*ast.CallExpr); ok {
			if identifier, ok := call.Fun.(*ast.Ident); ok && identifier.Name == name {
				return true
			}
		}
	}
	return false
}

func astContainsQualifiedCall(node ast.Node, info *types.Info, imports map[string]string, key string) bool {
	if node == nil {
		return false
	}
	for candidate := range ast.Preorder(node) {
		if call, ok := candidate.(*ast.CallExpr); ok && modernGoCallKey(call, info, imports) == key {
			return true
		}
	}
	return false
}

func astContainsSelector(node ast.Node, name string) bool {
	if node == nil {
		return false
	}
	for candidate := range ast.Preorder(node) {
		if selector, ok := candidate.(*ast.SelectorExpr); ok && selector.Sel.Name == name {
			return true
		}
	}
	return false
}

func modernGoBenchmarkLoopCandidate(node ast.Node, info *types.Info) bool {
	var expression ast.Expr
	switch loop := node.(type) {
	case *ast.ForStmt:
		expression = loop.Cond
	case *ast.RangeStmt:
		expression = loop.X
	default:
		return false
	}
	if expression == nil {
		return false
	}
	for candidate := range ast.Preorder(expression) {
		if selector, ok := candidate.(*ast.SelectorExpr); ok && modernGoBenchmarkCounter(selector, info) {
			return true
		}
	}
	return false
}

func modernGoBenchmarkCounter(selector *ast.SelectorExpr, info *types.Info) bool {
	if selector == nil || selector.Sel.Name != "N" {
		return false
	}
	if info == nil {
		receiver, ok := selector.X.(*ast.Ident)
		return ok && receiver.Name == "b"
	}
	object := info.Uses[selector.Sel]
	if object == nil {
		return false
	}
	return object.Pkg() != nil && object.Pkg().Path() == "testing"
}

func modernGoFallbackIf(statement *ast.IfStmt) bool {
	if statement == nil || statement.Init != nil || statement.Else != nil || len(statement.Body.List) != 1 {
		return false
	}
	comparison, ok := statement.Cond.(*ast.BinaryExpr)
	if !ok || comparison.Op != token.EQL {
		return false
	}
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return false
	}
	value := comparison.X
	if modernGoZeroValue(value) {
		value = comparison.Y
	} else if !modernGoZeroValue(comparison.Y) {
		return false
	}
	return expressionsEqual(assignment.Lhs[0], value)
}

func modernGoZeroValue(expression ast.Expr) bool {
	switch value := expression.(type) {
	case *ast.BasicLit:
		literal := constant.MakeFromLiteral(value.Value, value.Kind, uint(token.NoPos))
		switch literal.Kind() {
		case constant.String:
			return constant.StringVal(literal) == ""
		case constant.Int, constant.Float:
			return constant.Sign(literal) == int(token.NoPos)
		default:
			return false
		}
	case *ast.Ident:
		return value.Name == "false" || value.Name == "nil"
	default:
		return false
	}
}

func modernGoComparisonAssignment(statement *ast.IfStmt, info *types.Info) bool {
	_, _, _, ok := modernGoExtremaAssignment(statement, info)
	return ok
}

func modernGoExtremaAssignment(statement *ast.IfStmt, info *types.Info) (destination, other *ast.Ident, function string, ok bool) {
	if statement == nil || statement.Init != nil || statement.Else != nil || len(statement.Body.List) != 1 {
		return nil, nil, "", false
	}
	comparison, ok := statement.Cond.(*ast.BinaryExpr)
	if !ok || !containsModernGoCall([]string{"<", ">", "<=", ">="}, comparison.Op.String()) {
		return nil, nil, "", false
	}
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return nil, nil, "", false
	}
	left, leftOK := comparison.X.(*ast.Ident)
	right, rightOK := comparison.Y.(*ast.Ident)
	destination, destinationOK := assignment.Lhs[0].(*ast.Ident)
	other, otherOK := assignment.Rhs[0].(*ast.Ident)
	if !leftOK || !rightOK || !destinationOK || !otherOK {
		return nil, nil, "", false
	}
	destinationOnLeft := destination.Name == left.Name && other.Name == right.Name
	destinationOnRight := destination.Name == right.Name && other.Name == left.Name
	if !destinationOnLeft && !destinationOnRight {
		return nil, nil, "", false
	}
	if info != nil && (!modernGoOrderedType(info.TypeOf(destination)) || !types.Identical(info.TypeOf(destination), info.TypeOf(other))) {
		return nil, nil, "", false
	}
	function = "min"
	if destinationOnLeft && (comparison.Op == token.LSS || comparison.Op == token.LEQ) ||
		destinationOnRight && (comparison.Op == token.GTR || comparison.Op == token.GEQ) {
		function = "max"
	}
	return destination, other, function, true
}

func modernGoOrderedType(value types.Type) bool {
	if value == nil {
		return false
	}
	basic, ok := value.Underlying().(*types.Basic)
	return ok && basic.Info()&types.IsOrdered != 0
}

func modernGoSliceSearch(loop *ast.RangeStmt, rule string) bool {
	value, ok := loop.Value.(*ast.Ident)
	if !ok || value.Name == "_" || len(loop.Body.List) != 1 {
		return false
	}
	key, hasKey := loop.Key.(*ast.Ident)
	hasKey = hasKey && key.Name != "_"
	statement, ok := loop.Body.List[0].(*ast.IfStmt)
	if !ok || statement.Else != nil || len(statement.Body.List) != 1 || !astContainsIdent(statement.Cond, value.Name) {
		return false
	}
	returned, ok := statement.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(returned.Results) != 1 {
		return false
	}
	directEquality := modernGoDirectEquality(statement.Cond, value.Name)
	result, resultOK := returned.Results[0].(*ast.Ident)
	switch rule {
	case "slices_contains":
		return resultOK && result.Name == "true" && directEquality &&
			(!hasKey || !astContainsIdent(loop.Body, key.Name))
	case "slices_index":
		return resultOK && hasKey && result.Name == key.Name && directEquality
	case "slices_index_func":
		return resultOK && hasKey && result.Name == key.Name && !directEquality
	default:
		return false
	}
}

func modernGoDirectEquality(expression ast.Expr, name string) bool {
	comparison, ok := expression.(*ast.BinaryExpr)
	return ok && comparison.Op == token.EQL &&
		(directIdent(comparison.X, name) || directIdent(comparison.Y, name))
}

func modernGoSliceMinMax(loop *ast.RangeStmt, info *types.Info) bool {
	value, ok := loop.Value.(*ast.Ident)
	if !ok || value.Name == "_" {
		return false
	}
	found := false
	ast.Inspect(loop.Body, func(node ast.Node) bool {
		statement, ok := node.(*ast.IfStmt)
		found = found || ok && modernGoComparisonAssignment(statement, info) && astContainsIdent(statement.Cond, value.Name)
		return !found
	})
	return found
}

func modernGoSliceReverseLoop(loop *ast.ForStmt, info *types.Info) bool {
	if loop == nil || info == nil || len(loop.Body.List) != 1 {
		return false
	}
	initial, initialOK := loop.Init.(*ast.AssignStmt)
	condition, conditionOK := loop.Cond.(*ast.BinaryExpr)
	post, postOK := loop.Post.(*ast.AssignStmt)
	swap, swapOK := loop.Body.List[0].(*ast.AssignStmt)
	if !initialOK || !conditionOK || !postOK || !swapOK || initial.Tok != token.DEFINE || post.Tok != token.ASSIGN ||
		!modernGoBinaryExpressions(initial.Lhs) || !modernGoBinaryExpressions(initial.Rhs) ||
		!modernGoBinaryExpressions(post.Lhs) || !modernGoBinaryExpressions(post.Rhs) ||
		condition.Op != token.LSS || !modernGoSliceReverseAssignment(swap) {
		return false
	}
	left, leftOK := initial.Lhs[0].(*ast.Ident)
	right, rightOK := initial.Lhs[1].(*ast.Ident)
	zero, zeroOK := initial.Rhs[0].(*ast.BasicLit)
	last, lastOK := initial.Rhs[1].(*ast.BinaryExpr)
	if !lastOK {
		return false
	}
	length, lengthOK := last.X.(*ast.CallExpr)
	if !lengthOK {
		return false
	}
	lengthName, lengthNameOK := length.Fun.(*ast.Ident)
	one, oneOK := last.Y.(*ast.BasicLit)
	if !leftOK || !rightOK || !zeroOK || zero.Value != "0" || last.Op != token.SUB ||
		!lengthNameOK || lengthName.Name != "len" || len(length.Args) != 1 || !oneOK || one.Value != "1" ||
		!directIdent(condition.X, left.Name) || !directIdent(condition.Y, right.Name) {
		return false
	}
	leftStep, leftStepOK := post.Rhs[0].(*ast.BinaryExpr)
	rightStep, rightStepOK := post.Rhs[1].(*ast.BinaryExpr)
	if !directIdent(post.Lhs[0], left.Name) || !directIdent(post.Lhs[1], right.Name) || !leftStepOK || !rightStepOK ||
		leftStep.Op != token.ADD || rightStep.Op != token.SUB || !directIdent(leftStep.X, left.Name) ||
		!directIdent(rightStep.X, right.Name) || !basicLiteral(leftStep.Y, "1") || !basicLiteral(rightStep.Y, "1") {
		return false
	}
	first, firstOK := swap.Lhs[0].(*ast.IndexExpr)
	second, secondOK := swap.Lhs[1].(*ast.IndexExpr)
	if !firstOK || !secondOK || !expressionsEqual(first.X, length.Args[0]) || !expressionsEqual(second.X, length.Args[0]) ||
		!directIdent(first.Index, left.Name) || !directIdent(second.Index, right.Name) {
		return false
	}
	typeOf := info.TypeOf(first.X)
	if typeOf == nil {
		return false
	}
	_, slice := typeOf.Underlying().(*types.Slice)
	return slice
}

func basicLiteral(expression ast.Expr, value string) bool {
	literal, ok := expression.(*ast.BasicLit)
	return ok && literal.Value == value
}

func modernGoSliceReverseAssignment(assignment *ast.AssignStmt) bool {
	if assignment == nil || assignment.Tok != token.ASSIGN {
		return false
	}
	right := slices.Clone(assignment.Rhs)
	slices.Reverse(right)
	seen, multiple := false, false
	equal := slices.EqualFunc(assignment.Lhs, right, func(left, right ast.Expr) bool {
		if seen {
			multiple = true
		}
		seen = true
		leftIndex, leftOK := left.(*ast.IndexExpr)
		rightIndex, rightOK := right.(*ast.IndexExpr)
		return leftOK && rightOK && modernGoSameIndex(leftIndex, rightIndex)
	})
	return multiple && equal
}

func modernGoSameIndex(left, right *ast.IndexExpr) bool {
	return left != nil && right != nil && expressionsEqual(left.X, right.X) && expressionsEqual(left.Index, right.Index)
}

func modernGoSliceCompact(loop *ast.RangeStmt) bool {
	if loop == nil || len(loop.Body.List) != 1 {
		return false
	}
	value, ok := loop.Value.(*ast.Ident)
	if !ok || value.Name == "_" {
		return false
	}
	condition, ok := loop.Body.List[0].(*ast.IfStmt)
	if !ok || condition.Init != nil || condition.Else != nil || len(condition.Body.List) != 1 {
		return false
	}
	assignment, ok := condition.Body.List[0].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return false
	}
	destination, ok := assignment.Lhs[0].(*ast.Ident)
	if !ok {
		return false
	}
	appendCall, ok := assignment.Rhs[0].(*ast.CallExpr)
	if !ok || appendCall.Ellipsis.IsValid() || !modernGoBinaryExpressions(appendCall.Args) ||
		!directIdent(appendCall.Fun, "append") || !directIdent(appendCall.Args[0], destination.Name) ||
		!directIdent(appendCall.Args[1], value.Name) {
		return false
	}
	return modernGoCompactCondition(condition.Cond, destination.Name, value.Name)
}

func modernGoBinaryExpressions(expressions []ast.Expr) bool {
	return len(expressions) > 1 && len(expressions[1:]) == 1
}

func modernGoCompactCondition(expression ast.Expr, destination, value string) bool {
	if binary, ok := expression.(*ast.BinaryExpr); ok && binary.Op == token.LOR {
		return modernGoEmptySliceCondition(binary.X, destination) && modernGoAdjacentDuplicateCondition(binary.Y, destination, value)
	}
	return modernGoAdjacentDuplicateCondition(expression, destination, value)
}

func modernGoEmptySliceCondition(expression ast.Expr, destination string) bool {
	binary, ok := expression.(*ast.BinaryExpr)
	if !ok || binary.Op != token.EQL || !basicLiteral(binary.Y, "0") {
		return false
	}
	length, ok := binary.X.(*ast.CallExpr)
	return ok && len(length.Args) == 1 && directIdent(length.Fun, "len") && directIdent(length.Args[0], destination)
}

func modernGoAdjacentDuplicateCondition(expression ast.Expr, destination, value string) bool {
	binary, ok := expression.(*ast.BinaryExpr)
	if !ok || binary.Op != token.NEQ {
		return false
	}
	return directIdent(binary.X, value) && modernGoLastSliceElement(binary.Y, destination) ||
		directIdent(binary.Y, value) && modernGoLastSliceElement(binary.X, destination)
}

func modernGoLastSliceElement(expression ast.Expr, destination string) bool {
	indexed, ok := expression.(*ast.IndexExpr)
	if !ok || !directIdent(indexed.X, destination) {
		return false
	}
	last, ok := indexed.Index.(*ast.BinaryExpr)
	if !ok || last.Op != token.SUB || !basicLiteral(last.Y, "1") {
		return false
	}
	length, ok := last.X.(*ast.CallExpr)
	return ok && len(length.Args) == 1 && directIdent(length.Fun, "len") && directIdent(length.Args[0], destination)
}

func directIdent(expression ast.Expr, name string) bool {
	identifier, ok := expression.(*ast.Ident)
	return ok && identifier.Name == name
}

func modernGoEquivalentExpression(left, right ast.Expr) bool {
	switch left := left.(type) {
	case *ast.Ident:
		right, ok := right.(*ast.Ident)
		return ok && left.Name == right.Name
	case *ast.BasicLit:
		right, ok := right.(*ast.BasicLit)
		return ok && left.Kind == right.Kind && left.Value == right.Value
	case *ast.CallExpr:
		right, ok := right.(*ast.CallExpr)
		if !ok || len(left.Args) != len(right.Args) || !modernGoEquivalentExpression(left.Fun, right.Fun) {
			return false
		}
		for index := range left.Args {
			if !modernGoEquivalentExpression(left.Args[index], right.Args[index]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func astContainsIdent(root ast.Node, name string) bool {
	if root == nil {
		return false
	}
	for node := range ast.Preorder(root) {
		if identifier, ok := node.(*ast.Ident); ok && identifier.Name == name {
			return true
		}
	}
	return false
}

func isReflectTypeToken(call *ast.CallExpr) bool {
	return call != nil && len(call.Args) == 1 && astContainsSelector(call.Args[0], "Elem")
}

func modernGoMultipleWrappedErrors(call *ast.CallExpr) bool {
	if call == nil || len(call.Args) == 0 || len(call.Args[1:]) <= 1 {
		return false
	}
	literal, ok := call.Args[0].(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return false
	}
	format, err := strconv.Unquote(literal.Value)
	if err != nil {
		return false
	}
	return modernGoFormatVerbCount(format, 'w') > 1
}

func modernGoFormatVerbCount(format string, wanted byte) int {
	const verbs = "vTtbcdoOqxXUeEfFgGspw"
	count := 0
	for offset := 0; offset < len(format); offset++ {
		if format[offset] != '%' {
			continue
		}
		offset++
		if offset >= len(format) {
			break
		}
		if format[offset] == '%' {
			continue
		}
		for ; offset < len(format); offset++ {
			if !strings.ContainsRune(verbs, rune(format[offset])) {
				continue
			}
			if format[offset] == wanted {
				count++
			}
			break
		}
	}
	return count
}

func selectorReceiverCall(call *ast.CallExpr, info *types.Info, imports map[string]string, key string) bool {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	return ok && callExpressionKey(selector.X, info, imports, key)
}

func expressionsEqual(left, right ast.Expr) bool {
	leftIdent, leftOK := left.(*ast.Ident)
	rightIdent, rightOK := right.(*ast.Ident)
	return leftOK && rightOK && leftIdent.Name == rightIdent.Name
}

func isNilSlice(expression ast.Expr) bool {
	array, ok := expression.(*ast.CompositeLit)
	return ok && len(array.Elts) == 0
}

func isZeroExpression(expression ast.Expr) bool {
	literal, ok := expression.(*ast.BasicLit)
	if ok {
		return literal.Value == "0" || literal.Value == "0.0"
	}
	identifier, ok := expression.(*ast.Ident)
	return ok && (identifier.Name == "nil" || identifier.Name == "false")
}
