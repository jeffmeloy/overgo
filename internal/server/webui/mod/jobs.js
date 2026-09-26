(function () {
  "use strict";
  window.overgo.workflowWorkspace({ id: "training-jobs", scope: "training", trend: (name) => name.endsWith("_loss"), renderEvidence: (host, operation, overgo) => overgo.trainingEvidence.renderOperation(host, operation, overgo), });
  window.overgo.workflowWorkspace({ id: "export-jobs", scope: "export" });
})();
