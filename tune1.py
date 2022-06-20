import optuna
def objective(trial):
 x = trial.suggest_float("x", -10, 10)
 return (x - 2) ** 2
if __name__ == "__main__":
 study = optuna.create_study(
 study_name="distributed-example",
storage="postgresql://optuna:lkm961296@101.101.210.70:6013/optuna", load_if_exists=True
 )
 study.optimize(objective, n_trials=100)


