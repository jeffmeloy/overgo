(function () {
  "use strict";
  window.overgo.workflowWorkspace({ id: "training-jobs", scope: "training", longRun: true, trend: (name) => name.endsWith("_loss"), preview: (host, request, overgo) => overgo.trainingPreview(host, request, overgo), renderEvidence: (host, operation, overgo) => overgo.trainingEvidence.renderOperation(host, operation, overgo), });
  window.overgo.workflowWorkspace({ id: "export-jobs", scope: "export" });
})();
