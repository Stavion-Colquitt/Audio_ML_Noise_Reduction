"""Training progress at a glance: py -3.12 progress.py   (also prints the MLflow link for the live run)"""
import os, mlflow
HERE = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
exp = mlflow.get_experiment_by_name("denoiser-v3")
runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
r = runs.iloc[0]; rid = r.run_id
c = mlflow.tracking.MlflowClient()
hist = sorted(c.get_metric_history(rid, "dev_sisdr_gain_db"), key=lambda m: m.step)
print(f"run {rid}  status {r.status}  epochs done {len(hist)}/{r['params.epochs']}")
for m in hist[-5:]: print(f"  epoch {m.step:3d}  dev SI-SDR gain {m.value:+.2f} dB")
print(f"\nMLflow: http://127.0.0.1:5000/#/experiments/{exp.experiment_id}/runs/{rid}/model-metrics")
